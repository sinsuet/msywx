# -*- coding: utf-8 -*-
# 最终版本: 包含PRB利用率预测、训练/测试模式切换、模型保存/加载、性能记录
# 从并行决策修改为序列决策，解决动作空间组合爆炸问题


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import copy
import math
import os


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
        scaled_reward = spectrum_efficiency / 10.0
        qos_penalty = 0
        if rates['satellite'] < self.satellite_qos_threshold:
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


# 【新增】用于序列决策的全新Actor网络
class SatTerrestrialActor_Sequential(nn.Module):
    """序列决策Actor网络"""

    def __init__(self, obs_dim, num_channels, num_users, hidden_dim):
        super(SatTerrestrialActor_Sequential, self).__init__()
        self.num_channels = num_channels
        self.num_users = num_users
        self.hidden_dim = hidden_dim  # 添加hidden_dim属性

        # 基础特征提取网络
        self.feature_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU()
        )

        # GRU核心，用于处理序列信息。它的隐藏状态会记住已分配用户的信息
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        # 信道选择头，只为当前用户输出N个信道的选择概率
        self.channel_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_channels)
        )

        # 功率分配网络保持不变，但在序列决策最后一步调用
        # 根据实际观测维度计算power_net的输入维度
        # 基础观测维度: 2*users_per_gnb + num_channels + users_per_gnb + num_gnbs
        # 动作维度: users_per_gnb
        # 总维度: 2*2 + 10 + 2 + 3 + 2 = 21
        self.power_net = nn.Sequential(
            nn.Linear(21, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, num_users), nn.Sigmoid()
        )

    def forward(self, obs, hidden_state):
        """
        前向传播现在处理序列中的一步
        obs: 当前决策步骤的观测 (batch_size, obs_dim)
        hidden_state: GRU的上一步隐藏状态 (1, batch_size, hidden_dim)
        """
        # 1. 提取当前观测的特征
        features = self.feature_net(obs).unsqueeze(1)  # -> (batch_size, 1, hidden_dim)

        # 2. GRU处理序列信息
        # gru_out: (batch_size, 1, hidden_dim)
        # new_hidden_state: (1, batch_size, hidden_dim)
        gru_out, new_hidden_state = self.gru(features, hidden_state)

        # 3. 决策头输出当前用户的信道选择logits
        channel_logits = self.channel_head(gru_out.squeeze(1))  # -> (batch_size, num_channels)

        return channel_logits, new_hidden_state


class SatTerrestrialAgent:
    def __init__(self, obs_dim, num_channels, num_users, hidden_dim, lr=1e-4, device='cpu'):
        # 【修改】使用新的序列Actor网络
        self.actor = SatTerrestrialActor_Sequential(obs_dim, num_channels, num_users, hidden_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.target_actor = copy.deepcopy(self.actor)
        self.epsilon = 0.9
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.05
        self.num_channels = num_channels
        self.num_users = num_users
        self.device = device
        # 【新增】存储观测维度和隐藏层维度，方便后续使用
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim

    # 【重写并完成】select_action 方法
    def select_action(self, base_obs, power_min, power_max, use_exploration=True):
        if use_exploration and random.random() < self.epsilon:
            channel_actions = np.random.randint(0, self.num_channels, self.num_users)
            power_actions = np.random.uniform(power_min, power_max, self.num_users)
            return channel_actions, power_actions

        with torch.no_grad():
            channel_actions = []
            # 初始化GRU的隐藏状态
            hidden_state = torch.zeros(1, 1, self.hidden_dim).to(self.device)
            base_obs_tensor = torch.FloatTensor(base_obs).unsqueeze(0).to(self.device)

            # 【新增】为序列开始准备一个“空的”上一步动作的one-hot嵌入
            # 它的维度是 num_channels
            prev_action_embedding = torch.zeros(1, self.num_channels).to(self.device)

            # 循环K次，为每个用户决策
            for user_idx in range(self.num_users):
                # 【新增】创建当前用户ID的one-hot嵌入
                user_id_embedding = torch.zeros(1, self.num_users).to(self.device)
                user_id_embedding[0, user_idx] = 1.0

                # 【核心修改】为当前决策步构建动态观测
                # current_obs = [基础观测, 当前用户ID, 上一步动作]
                current_obs = torch.cat([
                    base_obs_tensor,
                    user_id_embedding,
                    prev_action_embedding
                ], dim=1)

                # 调用Actor进行单步决策
                channel_logits, hidden_state = self.actor(current_obs, hidden_state)

                # 从logits中采样一个动作
                channel_probs = torch.softmax(channel_logits, dim=-1)
                channel_dist = torch.distributions.Categorical(channel_probs)
                action = channel_dist.sample().item()
                channel_actions.append(action)

                # ==========================================================
                # 【完成PASS部分】为下一次循环准备“上一步动作”的嵌入
                # ==========================================================
                prev_action_embedding = torch.zeros(1, self.num_channels).to(self.device)
                prev_action_embedding[0, action] = 1.0
                # ==========================================================

            # 在所有信道分配完成后，进行一次功率分配
            # power_net期望的输入维度是base_obs_dim + num_users
            # base_obs_dim是基础观测维度，num_users是信道动作数量
            final_obs_for_power = torch.cat([
                base_obs_tensor,  # (1, base_obs_dim)
                torch.FloatTensor(np.array(channel_actions)).unsqueeze(0).to(self.device)  # (1, num_users)
            ], dim=1)

            power_output = self.actor.power_net(final_obs_for_power)
            power_actions = power_output[0].cpu().numpy() * (power_max - power_min) + power_min

        if use_exploration:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return np.array(channel_actions), power_actions

    # ... soft_update方法保持不变 ...
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
    def __init__(self, config):
        self.config = config
        self.env = SatTerrestrialEnvironment(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 【核心修改】重新定义观测维度
        # 1. 计算不包含序列信息的基础观测维度
        base_obs_dim = (
                self.env.users_per_gnb * 2 +  # gnb_to_users, sat_to_users
                self.env.num_channels +  # satellite_channels
                self.env.users_per_gnb +  # prev_actions (现在作为base)
                self.env.num_gnbs  # prb_prediction
        )

        # 2. 新的观测维度 = 基础维度 + 当前用户ID嵌入维度 + 上一步动作嵌入维度
        # 我们使用 one-hot 编码，所以嵌入维度等于类别数
        self.local_obs_dim = (
                base_obs_dim +
                self.env.users_per_gnb +  # one-hot for current user ID
                self.env.num_channels  # one-hot for previous action
        )
        # ----------------------------------------------------

        # 但是critic网络使用的是基础观测维度，不包含序列信息
        self.global_obs_dim = base_obs_dim * self.env.num_gnbs

        # 创建智能体和评论家网络 (这部分代码无需修改)
        self.agents = []
        for _ in range(config['num_gnbs']):
            # 【修改】Agent初始化时使用新的obs_dim定义
            # 注意：如果选择在select_action中拼接动作，这里的obs_dim需要调整
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

    def hierarchical_decision(self, global_state, use_exploration=True):
        """分层决策 (增加探索开关)"""
        actions = {}
        powers = {}
        for gnb_idx, agent in enumerate(self.agents):
            local_obs = self.env.get_local_observation(gnb_idx, global_state)
            channel_actions, power_actions = agent.select_action(
                local_obs, self.env.power_min, self.env.power_max, use_exploration=use_exploration
            )
            actions[gnb_idx] = channel_actions
            powers[gnb_idx] = power_actions
        return actions, powers

    # 在 SatTerrestrialHPGAMARLFramework 类中
    # 【重写】train_step 函数
    def train_step(self):
        """训练步骤 (序列决策版本)"""
        if len(self.replay_buffer) < self.batch_size:
            return

        # 1. 从经验回放池中采样一个批次的数据
        sampled_data = self.replay_buffer.sample(self.batch_size)
        if sampled_data is None:
            return
        # states, next_states: list of global_state dicts
        # actions: list of action dicts, e.g., {gnb_idx: (channel_actions, power_actions)}
        # rewards, dones: list of scalars
        states, actions, rewards, next_states, dones = sampled_data

        # 2. 准备全局观测和奖励，用于Critic训练
        # 将字典形式的全局状态转换为扁平化的numpy数组
        # 注意：在训练时，我们不需要添加序列信息（用户ID和上一步动作）
        all_current_obs = np.array([
            np.concatenate([self.env.get_local_observation(g, s) for g in range(self.env.num_gnbs)])
            for s in states
        ])
        all_next_obs = np.array([
            np.concatenate([self.env.get_local_observation(g, s) for g in range(self.env.num_gnbs)])
            for s in next_states
        ])

        # 转换为Tensor并移动到指定设备
        global_obs_tensor = torch.FloatTensor(all_current_obs).to(self.device)
        next_global_obs_tensor = torch.FloatTensor(all_next_obs).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones_tensor = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # 3. 训练全局Critic网络
        with torch.no_grad():
            # 使用Target Critic计算下一个状态的Q值
            next_values = self.target_critic(next_global_obs_tensor)
            # 使用贝尔曼方程计算目标Q值
            target_values = rewards_tensor + self.gamma * next_values * (1 - dones_tensor)

        # 计算当前Critic网络输出的Q值
        current_values = self.global_critic(global_obs_tensor)
        # 计算MSE损失
        critic_loss = nn.MSELoss()(current_values, target_values)

        # Critic网络反向传播和优化
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.global_critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # 4. 计算Advantage，用于指导所有Actor的训练
        with torch.no_grad():
            # Advantage代表了当前动作相比于平均动作的好坏程度
            advantage = target_values - current_values

        # 5. 遍历每个智能体，训练其Actor网络
        for gnb_idx, agent in enumerate(self.agents):
            # 准备该智能体的局部观测和动作序列
            local_obs_list = [self.env.get_local_observation(gnb_idx, s) for s in states]
            base_obs_tensor = torch.FloatTensor(np.array(local_obs_list)).to(self.device)

            # 从批次数据中提取该智能体的信道动作序列
            # action_sequences shape: (batch_size, num_users)
            action_sequences = np.array([a[gnb_idx][0] for a in actions])
            channel_action_tensor = torch.LongTensor(action_sequences).to(self.device)

            # ----- 核心：重放序列决策过程 -----

            # 初始化GRU的隐藏状态
            # Shape: (num_layers, batch_size, hidden_dim)
            hidden_state = torch.zeros(1, self.batch_size, agent.actor.hidden_dim).to(self.device)

            total_log_probs = torch.zeros(self.batch_size, 1).to(self.device)

            # 为序列开始准备一个"空的"上一步动作的one-hot嵌入
            prev_action_embedding = torch.zeros(self.batch_size, self.env.num_channels).to(self.device)

            # 循环K次，K = num_users，即序列的长度
            for user_idx in range(agent.num_users):
                # 创建当前用户ID的one-hot嵌入
                user_id_embedding = torch.zeros(self.batch_size, self.env.users_per_gnb).to(self.device)
                user_id_embedding[:, user_idx] = 1.0

                # 构建当前决策步的观测
                current_obs = torch.cat([
                    base_obs_tensor,
                    user_id_embedding,
                    prev_action_embedding
                ], dim=1)

                # Actor进行一步前向传播
                # 输入：当前步的观测和上一步的隐藏状态
                # 输出：当前步的logits和新的隐藏状态
                logits, hidden_state = agent.actor(current_obs, hidden_state)

                # 获取当前步实际执行的动作
                # Shape: (batch_size, 1)
                actions_at_this_step = channel_action_tensor[:, user_idx].unsqueeze(1)

                # 计算这些动作的对数概率
                log_softmax_dist = torch.log_softmax(logits, dim=-1)
                # 使用gather从分布中选取实际执行动作的对数概率
                action_log_probs = log_softmax_dist.gather(1, actions_at_this_step)

                # 累加序列的对数概率
                total_log_probs += action_log_probs

                # 为下一次循环准备"上一步动作"的嵌入
                prev_action_embedding = torch.zeros(self.batch_size, self.env.num_channels).to(self.device)
                prev_action_embedding.scatter_(1, actions_at_this_step, 1.0)

            # ----- 序列重放结束 -----

            # 计算Actor损失。根据策略梯度定理，损失是 -Advantage * log_prob(action_sequence)
            # 我们使用.detach()来阻止advantage的梯度回流到critic网络
            actor_loss = - (advantage.detach() * total_log_probs).mean()

            # Actor网络反向传播和优化
            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 1.0)
            agent.actor_optimizer.step()

        # 6. 在所有网络更新完毕后，软更新Target网络
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
        'users_per_gnb': 2,
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

    MODEL_PATH = "best_model.pth"
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
        training_episodes = 500
        best_avg_reward = -float('inf')
        reward_history = []

        for episode in range(training_episodes):
            state = framework.reset_environment()
            episode_reward = 0

            for step in range(50):
                actions, powers = framework.hierarchical_decision(state)
                next_state, reward, _, _, _ = framework.env.step(actions, powers)
                # 对奖励进行裁剪

                episode_reward += reward

                combined_actions = {gnb_idx: (actions[gnb_idx], powers[gnb_idx]) for gnb_idx in actions}
                framework.replay_buffer.push(state, combined_actions, reward, next_state, False)

                if len(framework.replay_buffer) > config['batch_size']:
                    framework.train_step()

                state = next_state

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

