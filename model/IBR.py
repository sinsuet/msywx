import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import copy
import math
import os
import time
import datetime


# class GameTheory_IBR_Policy:
#     """
#     【修正版】基于非合作博弈的迭代最佳响应 (IBR) 策略。
#     """
#
#     def __init__(self, env, iterations=5):  # 迭代次数可以减少
#         self.env = env
#         self.iterations = iterations
#         print(f"策略模式: 已初始化 GameTheory IBR Policy (修正版, 迭代 {self.iterations} 轮)")
#
#     def select_actions(self, global_state, use_exploration=False):
#         # 1. 初始化：所有gNB采用随机策略
#         actions = {gnb_idx: np.random.randint(0, self.env.num_channels, self.env.users_per_gnb)
#                    for gnb_idx in range(self.env.num_gnbs)}
#         powers = {gnb_idx: np.full(self.env.users_per_gnb, self.env.power_max / 2)
#                   for gnb_idx in range(self.env.num_gnbs)}
#
#         # 获取当前时刻的完整信道信息，在迭代中保持不变
#         channel_gains = self.env.get_channel_gains()
#
#         # 2. 迭代最佳响应
#         for _ in range(self.iterations):
#             # 轮流更新每个gNB的策略
#             for gnb_idx in range(self.env.num_gnbs):
#                 best_action, best_power = self._best_response(gnb_idx, actions, powers, channel_gains)
#                 actions[gnb_idx] = best_action
#                 powers[gnb_idx] = best_power
#
#         return actions, powers
#
#     def _best_response(self, gnb_idx, current_actions, current_powers, channel_gains):
#         """
#         【重写】为 gnb_idx 计算一个更真实的最佳响应。
#         """
#         gains_sat_to_ntn, gains_sat_to_users, gains_gnb_to_users, gains_gnb_interference = channel_gains
#
#         best_utility = -float('inf')
#         best_local_actions = np.zeros(self.env.users_per_gnb, dtype=int)
#         best_local_powers = np.zeros(self.env.users_per_gnb)
#
#         # 为了简化本地优化，我们采用两步法：先贪婪定信道，再优化功率
#
#         # --- a) 贪婪且无冲突的信道分配 (避免小区内冲突) ---
#         # 获取该gNB下所有用户在所有信道上的增益
#         # 注意: 这里需要 get_full_terrestrial_gains 辅助函数
#         _, _, gains_gnb_to_users_all_ch, _ = self.env.channel_model.get_full_terrestrial_gains(self.env)
#         local_gains = gains_gnb_to_users_all_ch[gnb_idx, :, :]  # (users_per_gnb, num_channels)
#
#         # 使用一个简单的贪婪匹配算法，确保每个用户分配到唯一的最佳信道
#         temp_gains = local_gains.copy()
#         local_actions = np.zeros(self.env.users_per_gnb, dtype=int)
#         for _ in range(self.env.users_per_gnb):
#             # 找到整个增益矩阵中的最大值
#             user_idx, ch_idx = np.unravel_index(np.argmax(temp_gains), temp_gains.shape)
#             local_actions[user_idx] = ch_idx
#             # 将该用户和该信道从候选池中移除，避免重复分配
#             temp_gains[user_idx, :] = -1
#             temp_gains[:, ch_idx] = -1
#
#         # --- b) 在固定信道下，进行简单的功率控制 ---
#         # 这里的功率控制可以采用多种启发式，例如最大功率，或者按信道质量比例分配
#         # 我们采用最大功率策略，因为它最能体现“自私”玩家的特性
#         local_powers = np.full(self.env.users_per_gnb, self.env.power_max)
#
#         # 理论上，_best_response 应该穷举本地所有可能的(动作,功率)组合，
#         # 并在一个模拟的 calculate_sinr_and_rates 中计算自身效用。
#         # 上述两步法是一个计算上可行的、效果远好于之前版本的近似。
#
#         return local_actions, local_powers

class GameTheory_IBR_Policy:
    """
    【最终修正版】基于非合作博弈的迭代最佳响应 (IBR) 策略。
    """

    def __init__(self, env, iterations=5):
        self.env = env
        self.iterations = iterations
        print(f"策略模式: 已初始化 GameTheory IBR Policy (最终修正版, 迭代 {self.iterations} 轮)")

    def select_actions(self, global_state, use_exploration=False):
        # 1. 初始化随机策略
        actions = {gnb_idx: np.random.randint(0, self.env.num_channels, self.env.users_per_gnb)
                   for gnb_idx in range(self.env.num_gnbs)}
        powers = {gnb_idx: np.full(self.env.users_per_gnb, self.env.power_max / 2)
                  for gnb_idx in range(self.env.num_gnbs)}

        # 获取当前时刻的完整信道信息
        channel_gains_tuple = self.env.get_channel_gains()

        # 2. 迭代最佳响应
        for _ in range(self.iterations):
            for gnb_idx in range(self.env.num_gnbs):
                best_action, best_power = self._best_response(gnb_idx, actions, powers, channel_gains_tuple)
                actions[gnb_idx] = best_action
                powers[gnb_idx] = best_power

        return actions, powers

    def _best_response(self, gnb_idx, current_actions, current_powers, channel_gains_tuple):
        """
        【最终重写版】为 gnb_idx 计算一个能处理 K > N 情况的最佳响应。
        """
        # --- a) 准备数据 ---
        _, _, gains_gnb_to_users_all_ch, _ = self.env.channel_model.get_full_terrestrial_gains(self.env)
        local_gains = gains_gnb_to_users_all_ch[gnb_idx, :, :]  # Shape: (16, 10)

        num_users = self.env.users_per_gnb
        num_channels = self.env.num_channels

        local_actions = np.full(num_users, -1, dtype=int)  # 初始化动作为-1（未分配）
        unassigned_users = list(range(num_users))

        # --- b) 第一阶段：信道锚定 (为每个信道找一个最佳用户) ---
        temp_gains = local_gains.copy()

        # 遍历所有信道
        for ch_idx in range(num_channels):
            if not unassigned_users:  # 如果用户都分配完了，提前结束
                break

            # 在当前信道上，找到增益最大的、且尚未被分配的用户
            best_user_for_this_channel = -1
            max_gain = -1

            # 从尚未分配的用户中寻找
            for user_idx in unassigned_users:
                if temp_gains[user_idx, ch_idx] > max_gain:
                    max_gain = temp_gains[user_idx, ch_idx]
                    best_user_for_this_channel = user_idx

            if best_user_for_this_channel != -1:
                # 分配信道
                local_actions[best_user_for_this_channel] = ch_idx
                # 将该用户标记为已分配
                unassigned_users.remove(best_user_for_this_channel)

        # --- c) 第二阶段：剩余用户分配 ---
        # 经过第一阶段，还剩下 16 - 10 = 6 个用户在 unassigned_users 列表中
        for user_idx in unassigned_users:
            # 为这个剩余用户，找到一个对他来说最好的信道
            best_channel_for_this_user = np.argmax(local_gains[user_idx, :])
            local_actions[user_idx] = best_channel_for_this_user

        # --- d) 功率分配 ---
        # 仍然采用最大功率策略，以体现“自私”特性
        local_powers = np.full(num_users, self.env.power_max)

        return local_actions, local_powers
