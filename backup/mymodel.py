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

from env.myenv import SatTerrestrialEnvironment


class SatTerrestrialActor(nn.Module):
    """(此类无需修改)"""

    def __init__(self, obs_dim, num_channels, num_users, hidden_dim):
        super(SatTerrestrialActor, self).__init__()
        self.num_channels = num_channels
        self.num_users = num_users
        self.channel_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, num_users * num_channels),
        )
        self.power_net = nn.Sequential(
            nn.Linear(obs_dim + num_users, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, num_users), nn.Sigmoid()
        )

    def forward(self, obs):
        channel_logits = self.channel_net(obs).view(-1, self.num_users, self.num_channels)
        channel_probs = torch.softmax(channel_logits, dim=-1)
        channel_selection_for_power = channel_probs.argmax(dim=-1)
        power_input = torch.cat([obs, channel_selection_for_power.float()], dim=1)
        power_output = self.power_net(power_input)
        return channel_probs, power_output


class SatTerrestrialAgent:
    """(此类已修正)"""

    def __init__(self, obs_dim, num_channels, num_users, hidden_dim, lr=1e-4, device='cpu'):
        self.actor = SatTerrestrialActor(obs_dim, num_channels, num_users, hidden_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.target_actor = copy.deepcopy(self.actor)
        self.epsilon = 0.9
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.num_channels = num_channels
        self.num_users = num_users
        self.device = device

    def select_action(self, obs, power_min, power_max, use_exploration=True):
        if use_exploration and random.random() < self.epsilon:
            channel_actions = np.random.randint(0, self.num_channels, self.num_users)
            power_actions = np.random.uniform(power_min, power_max, self.num_users)
            return channel_actions, power_actions
        with torch.no_grad():
            # 【修正】将输入张量移动到正确的设备
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            channel_probs, power_output = self.actor(obs_tensor)
            channel_dist = torch.distributions.Categorical(channel_probs[0])
            channel_actions = channel_dist.sample().cpu().numpy()
            power_actions = power_output[0].cpu().numpy() * (power_max - power_min) + power_min
        if use_exploration:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return channel_actions, power_actions

    def soft_update(self, tau=0.01):
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


class ReplayBuffer:
    """(此类无需修改)"""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        if not self.buffer or len(self.buffer) < batch_size:
            return None
        batch = random.sample(self.buffer, batch_size)
        return zip(*batch)

    def __len__(self):
        return len(self.buffer)


# ==============================================================================
# 框架修改：增加模型保存/加载和训练/测试切换
# ==============================================================================
class SatTerrestrialHPGAMARLFramework:
    """星地融合HPGA-MARL框架 (增加模型保存/加载和模式切换)"""

    def __init__(self, config):
        self.config = config
        self.model_name = config['model_name']  # 模型名
        self.model_params = config['model_params'][self.model_name]
        self.env = SatTerrestrialEnvironment(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 确定观测维度
        # self.local_obs_dim = (
        #         self.env.users_per_gnb * 2 +
        #         self.env.num_channels +
        #         self.env.users_per_gnb +
        #         self.env.num_gnbs
        # )

        # 最大尺寸观测维度
        # 【修改】使用 max_users_per_gnb 来计算观测维度
        self.local_obs_dim = (
                config['max_users_per_gnb'] * 2 + # 使用 max
                self.env.num_channels +
                config['max_users_per_gnb'] +     # 使用 max
                self.env.num_gnbs
        )

        self.global_obs_dim = self.local_obs_dim * self.env.num_gnbs

        # 创建智能体和评论家网络
        self.agents = []
        for _ in range(config['num_gnbs']):
            # 【修正】将device信息传递给Agent
            agent = SatTerrestrialAgent(
                self.local_obs_dim,
                self.env.num_channels,
                # self.env.users_per_gnb,
                self.config['max_users_per_gnb'],
                self.model_params['actor_hidden_dim'],
                self.model_params['actor_lr'],
                device=self.device
            )
            self.agents.append(agent)

        self.global_critic = nn.Sequential(
            nn.Linear(self.global_obs_dim, self.model_params['critic_hidden_dim']), nn.ReLU(),
            nn.Linear(self.model_params['critic_hidden_dim'], self.model_params['critic_hidden_dim']), nn.ReLU(),
            nn.Linear(self.model_params['critic_hidden_dim'], 1)
        )

        # 将网络移动到指定设备
        for agent in self.agents:
            agent.actor.to(self.device)
            agent.target_actor.to(self.device)
        self.global_critic.to(self.device)

        self.critic_optimizer = optim.Adam(self.global_critic.parameters(), lr=self.model_params['critic_lr'])
        self.target_critic = copy.deepcopy(self.global_critic).to(self.device)

        # 经验回放
        self.replay_buffer = ReplayBuffer(config['buffer_size'])
        self.gamma = config['gamma']
        self.batch_size = config['batch_size']

    def save_models(self, filename="best_model.pth"):
        """【新增】保存模型参数"""
        checkpoint = {
            'global_critic': self.global_critic.state_dict(),
            'actors': [agent.actor.state_dict() for agent in self.agents]
        }
        torch.save(checkpoint, filename)
        # print(f"Models saved to {filename}")

    def load_models(self, filename="best_model.pth"):
        """【新增】加载模型参数"""
        if not os.path.exists(filename):
            print(f"Error: Model file not found at {filename}. Cannot run in test mode.")
            exit()

        checkpoint = torch.load(filename, map_location=self.device)
        self.global_critic.load_state_dict(checkpoint['global_critic'])
        self.target_critic.load_state_dict(checkpoint['global_critic'])
        for i, agent in enumerate(self.agents):
            agent.actor.load_state_dict(checkpoint['actors'][i])
            agent.target_actor.load_state_dict(checkpoint['actors'][i])
        print(f"Models loaded from {filename}")

    # 在 SatTerrestrialHPGAMARLFramework 类中

    # def hierarchical_decision(self, global_state, use_exploration=True):
    #     """分层决策 (增加耗时统计)"""
    #     actions = {}
    #     powers = {}
    #
    #     # 【新增】初始化一个字典，用于存储本次函数调用内部的耗时
    #     time_stats = {
    #         'get_observation_time': 0.0,
    #         'select_action_time': 0.0
    #     }
    #
    #     for gnb_idx, agent in enumerate(self.agents):
    #         # 1. 计时：获取局部观测
    #         start_time = time.time()
    #         local_obs = self.env.get_local_observation(gnb_idx, global_state)
    #         # time_stats['get_observation_time'] += time.time() - start_time
    #
    #         # 2. 计时：智能体决策 (神经网络前向传播)
    #         start_time = time.time()
    #         channel_actions, power_actions = agent.select_action(
    #             local_obs, self.env.power_min, self.env.power_max, use_exploration=use_exploration
    #         )
    #         # time_stats['select_action_time'] += time.time() - start_time
    #
    #         actions[gnb_idx] = channel_actions
    #         powers[gnb_idx] = power_actions
    #
    #     # 【修改】在返回时，额外返回耗时统计字典
    #     return actions, powers

    def hierarchical_decision(self, global_state, use_exploration=True):
        """【修改】分层决策，实现输出掩码"""
        actions = {}
        powers = {}
        current_users_per_gnb = self.env.users_per_gnb  # 获取当前环境的真实用户数
        # 【核心修改】从当前环境中获取真实的基站数
        current_gnbs = self.env.num_gnbs

        for gnb_idx in range(current_gnbs):
            # 1. 获取填充后的、固定长度的观测
            agent = self.agents[gnb_idx] # 从完整的agent列表中按索引获取
            local_obs = self.env.get_local_observation(gnb_idx, global_state, self.config['max_gnbs'], self.config['max_users_per_gnb'])

            # 2. 智能体网络输出一个固定长度 (max_users_per_gnb) 的决策
            full_channel_actions, full_power_actions = agent.select_action(
                local_obs, self.env.power_min, self.env.power_max, use_exploration=use_exploration
            )

            # 3. 【核心】进行输出掩码，只采用前 current_users_per_gnb 个有效决策
            actions[gnb_idx] = full_channel_actions[:current_users_per_gnb]
            powers[gnb_idx] = full_power_actions[:current_users_per_gnb]

        return actions, powers

    # 在 SatTerrestrialHPGAMARLFramework 类中
    # 【优化后的 train_step 函数】
    def train_step(self):
        """训练步骤 (优化版)"""
        if len(self.replay_buffer) < self.batch_size:
            return

        sampled_data = self.replay_buffer.sample(self.batch_size)
        if sampled_data is None: return
        states, actions, rewards, next_states, dones = sampled_data

        # ==============================================================================
        # 【优化核心】第一步：一次性计算并缓存所有需要的局部观测
        # ==============================================================================
        # 创建缓存列表，每个智能体一个
        cached_current_local_obs = [[] for _ in range(self.env.num_gnbs)]
        cached_next_local_obs = [[] for _ in range(self.env.num_gnbs)]

        # 【新增】从config中获取max_users_per_gnb的值
        max_users = self.config['max_users_per_gnb']
        max_gnbs = self.config['max_gnbs']

        for i in range(len(states)):
            for gnb_idx in range(self.env.num_gnbs):
                # 【修改】在调用时，传入第三个参数 max_users
                cached_current_local_obs[gnb_idx].append(self.env.get_local_observation(gnb_idx, states[i],max_gnbs, max_users))
                cached_next_local_obs[gnb_idx].append(
                    self.env.get_local_observation(gnb_idx, next_states[i], max_gnbs, max_users))

        # # 只循环一次batch_size，计算所有智能体的观测并存入缓存
        # for i in range(len(states)):
        #     for gnb_idx in range(self.env.num_gnbs):
        #         cached_current_local_obs[gnb_idx].append(self.env.get_local_observation(gnb_idx, states[i]))
        #         cached_next_local_obs[gnb_idx].append(self.env.get_local_observation(gnb_idx, next_states[i]))

        # 将缓存转换为Tensor
        # cached_current_local_obs_tensors[g] 的形状为 (batch_size, local_obs_dim)
        cached_current_local_obs_tensors = [torch.FloatTensor(np.array(obs)).to(self.device) for obs in
                                            cached_current_local_obs]
        cached_next_local_obs_tensors = [torch.FloatTensor(np.array(obs)).to(self.device) for obs in
                                         cached_next_local_obs]

        # ==============================================================================
        # 【优化核心】第二步：使用缓存数据，避免重复计算
        # ==============================================================================

        # 【优化】直接从缓存的局部观测拼接成全局观测，无需再调用get_local_observation
        global_obs_tensor = torch.cat(cached_current_local_obs_tensors, dim=1)
        next_global_obs_tensor = torch.cat(cached_next_local_obs_tensors, dim=1)

        # 奖励和完成状态的Tensor转换（这部分不变）
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones_tensor = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # --- Critic训练部分 (逻辑不变, 但数据来源已优化) ---
        with torch.no_grad():
            next_values = self.target_critic(next_global_obs_tensor)
            target_values = rewards_tensor + self.gamma * next_values * (1 - dones_tensor)

        current_values = self.global_critic(global_obs_tensor)
        critic_loss = nn.MSELoss()(current_values, target_values)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.global_critic.parameters(), 1.0)
        self.critic_optimizer.step()

        with torch.no_grad():
            advantage = target_values - current_values

        # --- Actor训练部分 ---
        for gnb_idx, agent in enumerate(self.agents):
            # 【优化】直接从缓存中获取此智能体的局部观测Tensor，无需再计算
            local_obs_tensor = cached_current_local_obs_tensors[gnb_idx]

            # 动作解包（这部分不变，但可以考虑在采样时就处理好以进一步优化）
            action_list = [a[gnb_idx] for a in actions]
            channel_action_tensor = torch.LongTensor(np.array([ch_pow_pair[0] for ch_pow_pair in action_list])).to(
                self.device)

            # Actor网络计算（逻辑不变）
            channel_probs, power_output = agent.actor(local_obs_tensor)
            log_probs = torch.log(channel_probs + 1e-8)
            action_log_probs = log_probs.gather(2, channel_action_tensor.unsqueeze(-1)).squeeze(-1)
            actor_loss = - (advantage.detach() * action_log_probs.mean(axis=1, keepdim=True)).mean()

            # Actor网络优化（逻辑不变）
            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 1.0)
            agent.actor_optimizer.step()

        # --- 软更新部分 (逻辑不变) ---
        self.soft_update_critic(tau=0.005)
        for agent in self.agents:
            agent.soft_update(tau=0.005)

    def soft_update_critic(self, tau=0.01):
        for target_param, param in zip(self.target_critic.parameters(), self.global_critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def reset_environment(self):
        self.env = SatTerrestrialEnvironment(self.config)
        initial_future_actions = {i: np.random.randint(0, self.env.num_channels, self.env.users_per_gnb) for i in
                                  range(self.env.num_gnbs)}
        ideal_future_utilization = self.env.get_prb_utilization(initial_future_actions)
        noise = np.random.normal(0, self.env.prediction_noise_std, size=ideal_future_utilization.shape)
        initial_prediction = np.clip(ideal_future_utilization + noise, 0, 1)
        initial_gains = self.env.get_channel_gains()
        initial_actions = {gnb_idx: np.zeros(self.env.users_per_gnb, dtype=int) for gnb_idx in
                           range(self.config['num_gnbs'])}
        initial_powers = {gnb_idx: np.ones(self.env.users_per_gnb) * self.env.power_min for gnb_idx in
                          range(self.config['num_gnbs'])}
        initial_rates = {
            'terrestrial': {gnb_idx: [0] * self.env.users_per_gnb for gnb_idx in range(self.config['num_gnbs'])},
            'satellite': 0}
        return self.env._build_state(initial_gains, initial_actions, initial_powers, initial_rates, initial_prediction,
                                     0, 0)



