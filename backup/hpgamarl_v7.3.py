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


class SatelliteOrbitModel:
    """LEO卫星轨道模型 (此类无需修改)"""

    def __init__(self, altitude=600, earth_radius=6371):
        self.h = altitude
        self.Ra = earth_radius
        self.lambda_param = (earth_radius + altitude) / earth_radius
        self.velocity = 7.5
        self.psi = 0

    def update_position(self, dt=0.02):
        delta_psi = self.velocity * dt / (self.Ra + self.h)
        self.psi += delta_psi
        return self.psi

    def get_distance_to_ground(self):
        phi = self.get_elevation_angle()
        # 修正公式
        val_inside_sqrt = self.Ra ** 2 * math.sin(phi) ** 2 + self.h ** 2 + 2 * self.Ra * self.h
        distance = math.sqrt(val_inside_sqrt) - self.Ra * math.sin(phi)
        return distance

    def get_elevation_angle(self):
        cos_psi = math.cos(self.psi)
        denominator = math.sqrt(max(1e-10, 1 - cos_psi ** 2))
        if abs(denominator) < 1e-10:
            phi = math.pi / 2 if cos_psi > 0 else 0
        else:
            numerator = cos_psi - 1 / self.lambda_param
            phi = math.atan(max(-1e10, min(1e10, numerator / denominator)))
        phi = max(0, min(math.pi / 2, abs(phi)))
        return phi


class ChannelModel:
    """信道模型 (此类无需修改)"""

    def __init__(self, frequency=17.5e9):
        self.frequency = frequency

    def free_space_loss(self, distance_km):
        safe_distance = np.maximum(distance_km, 1e-6)
        return 32.45 + 20 * np.log10(self.frequency / 1e9) + 20 * np.log10(safe_distance)

    def shadowed_rician_fading(self, size, ms=8, bs=0.129, omega_s=0.372):
        return np.random.exponential(1 / bs, size=size) * np.random.gamma(ms, omega_s / ms, size=size)

    def nakagami_fading(self, size, m=3, omega=1):
        return np.random.gamma(m, omega / m, size=size)

    def get_sat_to_ground_channel(self, distance_km, size, is_ntn=True):
        path_loss_db = self.free_space_loss(distance_km)
        path_loss_linear = 10 ** (-path_loss_db / 10)
        if is_ntn:
            fading = self.shadowed_rician_fading(size, 8, 0.129, 0.372)
        else:
            fading = self.shadowed_rician_fading(size, 5, 0.251, 0.279)
        return path_loss_linear * fading

    def get_terrestrial_channel(self, distance_km, path_loss_exp=3.5):
        safe_distance = np.maximum(distance_km, 1e-6)
        path_loss_linear = (1 / safe_distance) ** path_loss_exp
        fading = self.nakagami_fading(size=distance_km.shape)
        return path_loss_linear * fading


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

        self.prediction_noise_std = config.get('prediction_noise_std', 0.0)

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
        # scaled_reward = spectrum_efficiency / 10.0
        scaled_reward = spectrum_efficiency / 5.0
        qos_penalty = 0
        if rates['satellite'] < self.satellite_qos_threshold: # 打印self.satellite_qos_threshold
            qos_penalty += 1.0
        all_terrestrial_rates = np.array(list(rates['terrestrial'].values()))
        qos_violations = np.sum(all_terrestrial_rates < self.terrestrial_qos_threshold)
        qos_penalty += qos_violations * 0.5
        reward = scaled_reward - qos_penalty
        future_actions = {i: np.random.randint(0, self.num_channels, self.users_per_gnb) for i in range(self.num_gnbs)}
        ideal_future_utilization = self.get_prb_utilization(future_actions)
        noise = np.random.normal(0, self.prediction_noise_std, size=ideal_future_utilization.shape)
        noisy_prediction = np.clip(ideal_future_utilization + noise, 0, 1)
        next_state = self._build_state(current_channel_gains, actions, powers, rates, noisy_prediction, qos_violations,
                                       spectrum_efficiency)
        return next_state, reward, rates, qos_violations, spectrum_efficiency

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

    def get_local_observation(self, gnb_idx, global_state):
        gains = global_state['channel_gains']
        local_channels = gains['gnb_to_users'][gnb_idx, :]
        sat_interference_channels = gains['sat_to_users'][gnb_idx, :]
        obs_parts = [
            np.log1p(local_channels / 1e-12),
            np.log1p(sat_interference_channels / 1e-12),
            global_state['satellite_channels']
        ]
        prev_actions = global_state.get('prev_actions')
        if prev_actions and gnb_idx in prev_actions:
            obs_parts.append(np.array(prev_actions[gnb_idx]))
        else:
            obs_parts.append(np.zeros(self.users_per_gnb))
        prediction = global_state.get('prb_prediction')
        if prediction is not None:
            obs_parts.append(prediction)
        else:
            obs_parts.append(np.zeros(self.num_gnbs))
        return np.concatenate(obs_parts).astype(np.float32)


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
        self.epsilon_min = 0.001
        self.num_channels = num_channels
        self.num_users = num_users
        self.device = device  # 【新增】存储device

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
        self.env = SatTerrestrialEnvironment(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 确定观测维度
        self.local_obs_dim = (
                self.env.users_per_gnb * 2 +
                self.env.num_channels +
                self.env.users_per_gnb +
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
                self.env.users_per_gnb,
                config['actor_hidden_dim'],
                config['actor_lr'],
                device=self.device
            )
            self.agents.append(agent)

        self.global_critic = nn.Sequential(
            nn.Linear(self.global_obs_dim, config['critic_hidden_dim']), nn.ReLU(),
            nn.Linear(config['critic_hidden_dim'], config['critic_hidden_dim']), nn.ReLU(),
            nn.Linear(config['critic_hidden_dim'], 1)
        )

        # 将网络移动到指定设备
        for agent in self.agents:
            agent.actor.to(self.device)
            agent.target_actor.to(self.device)
        self.global_critic.to(self.device)

        self.critic_optimizer = optim.Adam(self.global_critic.parameters(), lr=config['critic_lr'])
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

    def hierarchical_decision(self, global_state, use_exploration=True):
        """分层决策 (增加耗时统计)"""
        actions = {}
        powers = {}

        # 【新增】初始化一个字典，用于存储本次函数调用内部的耗时
        time_stats = {
            'get_observation_time': 0.0,
            'select_action_time': 0.0
        }

        for gnb_idx, agent in enumerate(self.agents):
            # 1. 计时：获取局部观测
            start_time = time.time()
            local_obs = self.env.get_local_observation(gnb_idx, global_state)
            time_stats['get_observation_time'] += time.time() - start_time

            # 2. 计时：智能体决策 (神经网络前向传播)
            start_time = time.time()
            channel_actions, power_actions = agent.select_action(
                local_obs, self.env.power_min, self.env.power_max, use_exploration=use_exploration
            )
            time_stats['select_action_time'] += time.time() - start_time

            actions[gnb_idx] = channel_actions
            powers[gnb_idx] = power_actions

        # 【修改】在返回时，额外返回耗时统计字典
        return actions, powers, time_stats

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

        # 只循环一次batch_size，计算所有智能体的观测并存入缓存
        for i in range(len(states)):
            for gnb_idx in range(self.env.num_gnbs):
                cached_current_local_obs[gnb_idx].append(self.env.get_local_observation(gnb_idx, states[i]))
                cached_next_local_obs[gnb_idx].append(self.env.get_local_observation(gnb_idx, next_states[i]))

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


if __name__ == "__main__":
    # ==========================================================================
    # 【新增】主控制逻辑：训练或测试
    # ==========================================================================
    config = {
        'mode': 'train',  # 'train' 或 'test'
        'num_gnbs': 3,
        'users_per_gnb': 10,
        'num_channels': 10,
        'terrestrial_qos_mbps': 10,
        'satellite_qos_mbps': 20,
        'critic_hidden_dim': 256,
        'actor_hidden_dim': 128,
        'critic_lr': 1e-4,
        'actor_lr': 3e-4,
        'buffer_size': 500000,
        'gamma': 0.99,
        'batch_size': 256,
        'prediction_noise_std': 0.1
    }

    MODEL_PATH = "../best_maddpg_model.pth"
    REWARD_PATH = "../reward_history.csv"

    framework = SatTerrestrialHPGAMARLFramework(config)

    print(f"星地融合网络配置:")
    print(f"- 运行模式: {config['mode']}")
    print(f"- 设备: {framework.device}")
    print(f"- gNB数量: {config['num_gnbs']}")
    print(f"- 每个gNB用户数: {config['users_per_gnb']}")
    print(f"- 局部观测维度 (含预测): {framework.local_obs_dim}")
    print(f"- 全局观测维度 (含预测): {framework.global_obs_dim}")
    print(f"- PRB利用率预测噪声标准差: {config['prediction_noise_std']}")

    if config['mode'] == 'train':
        print("\n--- 开始训练 ---")
        training_episodes = 5000
        best_avg_reward = -float('inf')
        reward_history = []

        for episode in range(training_episodes):
            state = framework.reset_environment()
            episode_reward = 0

            # 【修改】在总的耗时统计字典中，为决策的内部细节增加条目
            time_stats = {
                'decision_total': 0.0,
                'decision_get_obs': 0.0,  # 新增：用于记录获取观测的时间
                'decision_select_action': 0.0,  # 新增：用于记录动作选择的时间
                'env_step': 0.0,
                'buffer_push': 0.0,
                'train_step': 0.0,
                'total_step_time': 0.0
            }

            for step in range(25):  # 为什么episode中还有个step循环？
                step_start_time = time.time()  # 记录单步总时间的开始
                # 1. 计时：决策过程
                start_time = time.time()

                # actions, powers = framework.hierarchical_decision(state)
                actions, powers, decision_times = framework.hierarchical_decision(state)

                time_stats['decision_total'] += time.time() - start_time
                # 【新增】将内部耗时累加到总的统计中
                time_stats['decision_get_obs'] += decision_times['get_observation_time']
                time_stats['decision_select_action'] += decision_times['select_action_time']

                # 2. 计时：环境交互
                start_time = time.time()
                next_state, reward, _, _, _ = framework.env.step(actions, powers)
                time_stats['env_step'] += time.time() - start_time


                episode_reward += reward

                # 3. 计时：存入经验池
                start_time = time.time()
                combined_actions = {gnb_idx: (actions[gnb_idx], powers[gnb_idx]) for gnb_idx in actions}
                framework.replay_buffer.push(state, combined_actions, reward, next_state, False)
                time_stats['buffer_push'] += time.time() - start_time

                # 4. 计时：模型训练
                start_time = time.time()
                if len(framework.replay_buffer) > config['batch_size']:
                    framework.train_step()
                time_stats['train_step'] += time.time() - start_time

                state = next_state

                time_stats['total_step_time'] += time.time() - step_start_time

            # 【新增】在一个episode结束后，打印平均耗时统计
            print(f"--- Episode {episode} 耗时分析 (平均每步) ---")
            steps_in_episode = 50
            for key, total_time in time_stats.items():
                avg_time = total_time / steps_in_episode
                # 打印毫秒(ms)为单位的时间，更直观
                print(f"- {key:<15}: {avg_time * 1000:.4f} ms")
            print("------------------------------------")

            avg_reward = episode_reward / 50
            reward_history.append(avg_reward)

            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                framework.save_models(MODEL_PATH)
                print(f"*** Episode {episode}: New best reward {best_avg_reward:.2f}, model saved. ***")

            if episode % 100 == 0:
                avg_100 = np.mean(reward_history[-100:]) if reward_history else 0.0
                print(f"Episode {episode}: Avg Reward(100) = {avg_100:.2f}, Best Avg Reward = {best_avg_reward:.2f}")

        np.savetxt(REWARD_PATH, np.array(reward_history), delimiter=",")
        print(f"\n训练完成! Reward历史已保存到 {REWARD_PATH}")

    elif config['mode'] == 'test':
        print("\n--- 开始测试 ---")
        framework.load_models(MODEL_PATH)

        test_episodes = 500
        total_test_reward = 0
        total_test_se = 0
        total_qos_violations = 0

        for episode in range(test_episodes):
            state = framework.reset_environment()
            episode_reward = 0

            for step in range(50):
                # 【关键】在测试时关闭探索
                actions, powers = framework.hierarchical_decision(state, use_exploration=False)
                next_state, reward, _, qos_violations, spectrum_efficiency = framework.env.step(actions, powers)

                total_test_reward += reward
                total_test_se += spectrum_efficiency
                total_qos_violations += qos_violations

                state = next_state

            if (episode + 1) % 100 == 0:
                print(f"已完成 {episode + 1}/{test_episodes} 个测试回合...")

        # 计算并打印平均性能
        avg_reward = total_test_reward / (test_episodes * 50)
        avg_se = total_test_se / (test_episodes * 50)
        avg_qos_violations = total_qos_violations / (test_episodes * 50)

        print("\n--- 测试结果 ---")
        print(f"平均奖励: {avg_reward:.4f}")
        print(f"平均频谱效率 (bps/Hz): {avg_se:.4f}")
        print(f"平均每步QoS违规用户数: {avg_qos_violations:.4f}")

