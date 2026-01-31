# -*- coding: utf-8 -*-
# 最终版本: 包含PRB利用率预测、训练/测试模式切换、模型保存/加载、性能记录

# 多分枝决策

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

from physic.physic import SatelliteOrbitModel,ChannelModel

class SatTerrestrialEnvironment:
    """星地融合频谱共享环境 (此类无需修改)"""

    def __init__(self, config):
        self.config = config
        self.satellite = SatelliteOrbitModel()
        self.channel_model = ChannelModel()

        # 网络拓扑参数
        self.num_gnbs = config['num_gnbs']


        self.num_channels = config['num_channels']
        self.users_per_gnb = config['users_per_gnb']
        self.total_bandwidth = 200e6
        self.channel_bandwidth = self.total_bandwidth / self.num_channels

        self.prediction_noise_std = config.get('prediction_noise_std', 0.0)  # 作为环境对象的一个属性

        # QoS要求
        self.terrestrial_qos_threshold = config.get('terrestrial_qos_mbps', 2)
        self.satellite_qos_threshold = config.get('satellite_qos_mbps', 5)

        # 功率常量
        self.power_min = 10 ** (15 / 10) / 1000
        self.power_max = 10 ** (45 / 10) / 1000
        self.sat_power = 1.0
        self.noise_power = 1e-12

        # 初始化位置
        self.ntn_position = np.array([0, 0])
        self.gnb_positions = self._generate_gnb_positions()
        self.user_positions = self._generate_user_positions()

        # 卫星信道占用
        self.satellite_channel_allocation = np.random.choice(
            [0, 1], size=self.num_channels, p=[0.6, 0.4]
        )

        self._precompute_distances()
        self._cache_static_channel_gains()

    def _precompute_distances(self):
        self.distances_gnb_to_user = np.linalg.norm(
            self.gnb_positions[:, np.newaxis, :] - self.user_positions, axis=2
        )
        # gNB 之间的距离矩阵，用作拓扑先验
        self.gnb_distance_matrix = np.linalg.norm(
            self.gnb_positions[:, np.newaxis, :] - self.gnb_positions[np.newaxis, :, :],
            axis=2
        )
        self.gnb_max_distance = max(1e-6, float(np.max(self.gnb_distance_matrix)))
        self.distances_inter_gnb = np.linalg.norm(
            self.user_positions[:, :, np.newaxis, :] - self.gnb_positions[np.newaxis, np.newaxis, :, :],
            axis=3
        )

    def _generate_gnb_positions(self):
        if self.num_gnbs == 1:
            return np.array([[5, 0]])
        num_to_generate = min(6, self.num_gnbs)
        angles = np.linspace(0, 2 * np.pi * (num_to_generate - 1) / num_to_generate, num_to_generate)
        positions = 8 * np.vstack([np.cos(angles), np.sin(angles)]).T
        if self.num_gnbs > 6:
            num_extra = self.num_gnbs - 6
            extra_angles = np.random.uniform(0, 2 * np.pi, num_extra)
            extra_radii = np.random.uniform(10, 15, num_extra)
            extra_pos = extra_radii[:, np.newaxis] * np.vstack([np.cos(extra_angles), np.sin(extra_angles)]).T
            positions = np.vstack([positions, extra_pos])
        return positions

    def _generate_user_positions(self):
        num_users_total = self.num_gnbs * self.users_per_gnb
        angles = np.random.uniform(0, 2 * np.pi, num_users_total)
        radii = np.random.uniform(0.5, 3, num_users_total)
        offsets = radii[:, np.newaxis] * np.vstack([np.cos(angles), np.sin(angles)]).T
        offsets = offsets.reshape(self.num_gnbs, self.users_per_gnb, 2)
        return self.gnb_positions[:, np.newaxis, :] + offsets

    def _cache_static_channel_gains(self):
        self.cached_gains_gnb_to_users = self.channel_model.get_terrestrial_channel(self.distances_gnb_to_user)
        gains_gnb_interference = self.channel_model.get_terrestrial_channel(self.distances_inter_gnb)
        for k in range(self.users_per_gnb):
            np.fill_diagonal(gains_gnb_interference[:, k, :], 0)
        self.cached_gains_gnb_interference = gains_gnb_interference

    def get_channel_gains(self, for_future=False):
        if for_future:
            future_satellite = copy.deepcopy(self.satellite)
            future_satellite.update_position()
            sat_distance = future_satellite.get_distance_to_ground()
        else:
            sat_distance = self.satellite.get_distance_to_ground()
        gains_sat_to_ntn = self.channel_model.get_sat_to_ground_channel(sat_distance, size=1, is_ntn=True)
        gains_sat_to_users = self.channel_model.get_sat_to_ground_channel(
            sat_distance, size=(self.num_gnbs, self.users_per_gnb), is_ntn=False
        )
        return gains_sat_to_ntn, gains_sat_to_users, self.cached_gains_gnb_to_users, self.cached_gains_gnb_interference

    def calculate_sinr_and_rates(self, channel_gains_tuple, actions, powers):
        gains_sat_to_ntn, gains_sat_to_users, gains_gnb_to_users, gains_gnb_interference = channel_gains_tuple
        actions_arr = np.array(list(actions.values()))
        powers_arr = np.array(list(powers.values()))
        sat_signal = self.sat_power * gains_sat_to_ntn
        is_interfering_on_sat_ch = self.satellite_channel_allocation[actions_arr] == 1
        sat_interference = np.sum(powers_arr[is_interfering_on_sat_ch] * gains_sat_to_users[is_interfering_on_sat_ch])
        sat_sinr = sat_signal / (sat_interference + self.noise_power)
        satellite_rate = self.channel_bandwidth * np.log2(1 + sat_sinr) / 1e6
        signal_power = powers_arr * gains_gnb_to_users
        sat_interference_on_users = self.sat_power * gains_sat_to_users * is_interfering_on_sat_ch
        channel_mask = actions_arr[..., np.newaxis] == np.arange(self.num_channels)
        power_on_channel = powers_arr[..., np.newaxis] * channel_mask
        power_sum_per_gnb_ch = np.sum(power_on_channel, axis=1)
        victim_channels = actions_arr
        interfering_powers = power_sum_per_gnb_ch[:, victim_channels]
        interfering_powers = np.transpose(interfering_powers, (1, 2, 0))
        interference_products = interfering_powers * gains_gnb_interference
        inter_cell_interference = np.sum(interference_products, axis=2)
        intra_cell_power_sum_ch = np.sum(power_on_channel, axis=1, keepdims=True)
        noma_interfering_power = intra_cell_power_sum_ch - power_on_channel
        noma_interference_for_user = np.sum(noma_interfering_power * channel_mask, axis=2)
        noma_interference = 0.1 * noma_interference_for_user * gains_gnb_to_users
        total_interference = sat_interference_on_users + inter_cell_interference + noma_interference
        terrestrial_sinr = signal_power / (total_interference + self.noise_power)
        terrestrial_rates_arr = self.channel_bandwidth * np.log2(1 + terrestrial_sinr) / 1e6
        rates = {'satellite': satellite_rate.item(), 'terrestrial': {}}
        for i in range(self.num_gnbs):
            rates['terrestrial'][i] = terrestrial_rates_arr[i].tolist()
        return rates

    def get_prb_utilization(self, actions):
        actions_arr = np.array(list(actions.values()))
        utilized_channels_per_gnb = [len(np.unique(actions_arr[i])) for i in range(self.num_gnbs)]
        return np.array(utilized_channels_per_gnb) / self.num_channels


    def step(self, actions, powers):
        self.satellite.update_position()
        current_channel_gains = self.get_channel_gains()
        rates = self.calculate_sinr_and_rates(current_channel_gains, actions, powers)


        spectrum_efficiency = np.sum(list(rates['terrestrial'].values())) + rates['satellite']
        spectrum_efficiency /= (self.total_bandwidth / 1e6)
        # 归一化到每个 gNB，避免 num_gnbs 放大奖励尺度
        spectrum_efficiency /= max(1, self.num_gnbs)

        # 【修改】奖励函数：使用正向激励代替负向惩罚
        # 基础奖励：归一化的频谱效率 (0~3 范围)
        # 进一步按用户规模做温和归一化，避免大规模奖励信号过弱或过强
        reference_total_users = 7 * 16
        current_total_users = max(1, self.num_gnbs * self.users_per_gnb)
        normalization_factor = np.sqrt(current_total_users / reference_total_users)
        base_reward = spectrum_efficiency / (5.0 * normalization_factor)

        # 正向奖励：QoS 满足时加分
        reward = base_reward

        if rates['satellite'] >= self.satellite_qos_threshold:
            reward += 0.5  # 卫星QoS满足奖励

        # 地面QoS满足率奖励
        all_terrestrial_rates = np.array(list(rates['terrestrial'].values()))
        qos_violations = np.sum(all_terrestrial_rates < self.terrestrial_qos_threshold)
        terrestrial_qos_satisfied = np.sum(all_terrestrial_rates >= self.terrestrial_qos_threshold)
        terrestrial_qos_ratio = terrestrial_qos_satisfied / all_terrestrial_rates.size
        reward += terrestrial_qos_ratio * 0.5  # 最大加0.5

        # 拥塞惩罚（可选，如果需要保留）
        congestion_penalty = 0.0
        w_congestion = 0.1
        prediction = self._get_next_prediction()
        actions_arr = np.array(list(actions.values()))

        for gnb_idx in range(self.num_gnbs):
            num_unique_channels = len(np.unique(actions_arr[gnb_idx]))
            congestion_penalty += w_congestion * prediction[gnb_idx] * (num_unique_channels / self.num_channels)

        # 按 gNB 数量归一化惩罚，避免规模放大
        congestion_penalty /= max(1, self.num_gnbs)
        reward = reward - congestion_penalty

        # 远见奖励/惩罚
        foresight_score = 0.0
        w_foresight = 0.05

        actions_arr = np.array(list(actions.values()))
        for gnb_idx in range(self.num_gnbs):
            if prediction[gnb_idx] > 0.7:
                num_unique_channels = len(np.unique(actions_arr[gnb_idx]))
                foresight_score -= (num_unique_channels / self.num_channels)

        # 按 gNB 数量归一化前瞻性惩罚
        foresight_score /= max(1, self.num_gnbs)
        reward = reward + w_foresight * foresight_score

        # 额外奖励缩放：用于大规模场景稳定训练
        reward_scale_factor = float(self.config.get('reward_scale_factor', 1.0))
        reward *= reward_scale_factor

        # future_actions = {i: np.random.randint(0, self.num_channels, self.users_per_gnb) for i in range(self.num_gnbs)}
        # ideal_future_utilization = self.get_prb_utilization(future_actions)
        # noise = np.random.normal(0, self.prediction_noise_std, size=ideal_future_utilization.shape)
        # noisy_prediction = np.clip(ideal_future_utilization + noise, 0, 1)
        # next_state = self._build_state(current_channel_gains, actions, powers, rates, noisy_prediction, qos_violations,
        #                                spectrum_efficiency)

        # 1. 获取下一时刻的信道增益作为“未来”的依据
        # 这里的 get_full_terrestrial_gains 是我们之前为 GDFP 策略添加的方法
        _, _, future_gains_all_ch, _ = self.channel_model.get_full_terrestrial_gains(self)

        # 2. 模拟未来的贪婪动作：每个用户都选择自己增益最高的信道
        future_actions = {}
        for gnb_idx in range(self.num_gnbs):
            gnb_gains = future_gains_all_ch[gnb_idx, :, :]  # (users_per_gnb, num_channels)
            future_actions[gnb_idx] = np.argmax(gnb_gains, axis=1)

        # 3. 基于这个更真实的 future_actions 计算利用率
        ideal_future_utilization = self.get_prb_utilization(future_actions)

        # 4. 添加噪声，模拟外部预测模块的准确率
        noise = np.random.normal(0, self.prediction_noise_std, size=ideal_future_utilization.shape)
        noisy_prediction = np.clip(ideal_future_utilization + noise, 0, 1)

        # 将这个高质量的预测值打包到 next_state 中
        next_state = self._build_state(current_channel_gains, actions, powers, rates,
                                       prediction=noisy_prediction,
                                       qos_violations=qos_violations,
                                       spectrum_efficiency=spectrum_efficiency)
        return next_state, reward, rates, qos_violations, spectrum_efficiency

    # def step(self, actions, powers):
    #     self.satellite.update_position()
    #     current_channel_gains = self.get_channel_gains()
    #     rates = self.calculate_sinr_and_rates(current_channel_gains, actions, powers)
    #     spectrum_efficiency = np.sum(list(rates['terrestrial'].values())) + rates['satellite']
    #     spectrum_efficiency /= (self.total_bandwidth / 1e6)
    #     # scaled_reward = spectrum_efficiency / 10.0
    #     scaled_reward = spectrum_efficiency / 5.0
    #     qos_penalty = 0
    #     if rates['satellite'] < self.satellite_qos_threshold: # 打印self.satellite_qos_threshold
    #         qos_penalty += 1.0
    #     all_terrestrial_rates = np.array(list(rates['terrestrial'].values()))
    #     qos_violations = np.sum(all_terrestrial_rates < self.terrestrial_qos_threshold)
    #     qos_penalty += qos_violations * 0.5
    #     reward = scaled_reward - qos_penalty
    #     future_actions = {i: np.random.randint(0, self.num_channels, self.users_per_gnb) for i in range(self.num_gnbs)}
    #     ideal_future_utilization = self.get_prb_utilization(future_actions)
    #     noise = np.random.normal(0, self.prediction_noise_std, size=ideal_future_utilization.shape)
    #     noisy_prediction = np.clip(ideal_future_utilization + noise, 0, 1)
    #     next_state = self._build_state(current_channel_gains, actions, powers, rates, noisy_prediction, qos_violations,
    #                                    spectrum_efficiency)
    #     return next_state, reward, rates, qos_violations, spectrum_efficiency

    # 您需要一个辅助函数来获取下一时隙的预测
    def _get_next_prediction(self):
        # 这里的逻辑应与 step 函数中生成 prediction 的逻辑一致
        _, _, future_gains_all_ch, _ = self.channel_model.get_full_terrestrial_gains(self)
        future_actions = {}
        for gnb_idx in range(self.num_gnbs):
            gnb_gains = future_gains_all_ch[gnb_idx, :, :]
            future_actions[gnb_idx] = np.argmax(gnb_gains, axis=1)
        ideal_future_utilization = self.get_prb_utilization(future_actions)
        return ideal_future_utilization  # 返回没有加噪声的理想值

    def reset(self):
        """
        重置环境到一个新的初始状态，并返回一个合法的 state 字典。
        """
        # 重新初始化环境，以获得全新的随机位置和信道等
        self.__init__(self.config)

        # 【核心修正】正确地为 prediction 创建一个全零的、维度正确的numpy数组
        initial_prediction = np.zeros(self.num_gnbs)

        # 使用关键字参数调用 _build_state，确保不会错位，并传入合法的 initial_prediction
        initial_state = self._build_state(
            channel_gains_tuple=self.get_channel_gains(),
            actions={},
            powers={},
            rates={},
            prediction=initial_prediction,  # <- 确保这里传入的是数组，而不是None
            qos_violations=0,
            spectrum_efficiency=0
        )
        return initial_state

    def _build_state(self, channel_gains_tuple, actions, powers, rates, prediction, qos_violations,
                     spectrum_efficiency):
        gains_sat_to_ntn, gains_sat_to_users, gains_gnb_to_users, gains_gnb_interference = channel_gains_tuple
        return {
            'channel_gains': {
                'sat_to_users': gains_sat_to_users,
                'gnb_to_users': gains_gnb_to_users,
            },
            'satellite_channels': self.satellite_channel_allocation,
            'prev_actions': actions,
            'prb_prediction': prediction,
        }


    def get_local_observation(self, gnb_idx, global_state, max_gnbs, max_users_per_gnb):
        current_users_per_gnb = self.users_per_gnb

        # --- 1. 处理用户相关的、需要填充的部分 ---
        gains = global_state['channel_gains']

        # 初始化固定长度的填充向量
        padded_local_channels = np.zeros(max_users_per_gnb)
        padded_sat_interference = np.zeros(max_users_per_gnb)
        padded_prev_actions = np.zeros(max_users_per_gnb)

        # 获取真实数据
        real_local_channels = gains['gnb_to_users'][gnb_idx, :]
        real_sat_interference = gains['sat_to_users'][gnb_idx, :]

        # 填充真实数据到向量的开头
        padded_local_channels[:current_users_per_gnb] = real_local_channels
        padded_sat_interference[:current_users_per_gnb] = real_sat_interference

        # 安全地处理 prev_actions
        prev_actions_dict = global_state.get('prev_actions', {})
        if gnb_idx in prev_actions_dict:
            actions_tuple = prev_actions_dict[gnb_idx]
            real_prev_actions = np.array(actions_tuple[0]) if isinstance(actions_tuple, tuple) else np.array(
                actions_tuple)
            if real_prev_actions.shape[0] == current_users_per_gnb:
                padded_prev_actions[:current_users_per_gnb] = real_prev_actions

        # --- 2. 处理全局的、无需填充的部分 ---

        # a) 卫星信道占用 (长度为 num_channels)
        satellite_channels = global_state['satellite_channels']
        # b) 资源利用率预测 (长度为 num_gnbs)
        # prediction = global_state.get('prb_prediction', np.zeros(self.num_gnbs))
        # a) 获取真实长度的 prediction 向量 (长度为 current_gnbs)
        prediction = global_state.get('prb_prediction', np.zeros(self.num_gnbs))
        prediction = np.atleast_1d(prediction)
        # b) 创建一个固定长度 (max_gnbs) 的零向量
        padded_prediction = np.zeros(max_gnbs)
        # c) 将真实数据填充到开头
        padded_prediction[:self.num_gnbs] = prediction

        # ==========================================================
        # 【核心修正】确保 prediction 是一个完整的数组，并且至少是一维的
        # ==========================================================
        prediction_vector = np.atleast_1d(padded_prediction)
        # ==========================================================

        # # --- 3. 拼接所有部分形成最终的观测向量 ---
        # obs_parts = [
        #     np.log1p(padded_local_channels / 1e-12),  # 长度: max_users_per_gnb
        #     np.log1p(padded_sat_interference / 1e-12),  # 长度: max_users_per_gnb
        #     satellite_channels,  # 长度: num_channels
        #     padded_prev_actions,  # 长度: max_users_per_gnb
        #     prediction_vector  # 长度: max_users_per_gnb
        # ]
        #
        # return np.concatenate(obs_parts).astype(np.float32)
        # 简化版：我们只调整与 gnb 相关的特征
        # 这里的 padded_prediction 长度是 max_gnbs
        gnb_congestion_prediction = padded_prediction[gnb_idx]

        # 创建新特征：将信道增益乘以 (1 - 拥塞度)
        congestion_adjusted_channels = padded_local_channels * (1 - gnb_congestion_prediction)
        # ==========================================================

        # gNB 拓扑距离（归一化，长度为 max_gnbs）
        gnb_distances = self.gnb_distance_matrix[gnb_idx]
        padded_gnb_distances = np.zeros(max_gnbs)
        padded_gnb_distances[:self.num_gnbs] = gnb_distances / self.gnb_max_distance

        obs_parts = [
            # 【修改】使用新特征替换或补充旧特征
            np.log1p(padded_local_channels / 1e-12),
            np.log1p(congestion_adjusted_channels / 1e-12),  # <-- 新增的特征
            np.log1p(padded_sat_interference / 1e-12),
            satellite_channels,
            padded_prev_actions,
            padded_prediction,
            padded_gnb_distances
        ]

        # print(np.concatenate(obs_parts).astype(np.float32).shape)

        return np.concatenate(obs_parts).astype(np.float32)





