# -*- coding: utf-8 -*-
# 完整矢量化修改

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GATConv
import random
from collections import deque
import copy
import math
import numba


class SatelliteOrbitModel:
    """LEO卫星轨道模型"""

    def __init__(self, altitude=600, earth_radius=6371):
        self.h = altitude  # km
        self.Ra = earth_radius  # km
        self.lambda_param = (earth_radius + altitude) / earth_radius
        self.velocity = 7.5  # km/s, typical LEO velocity
        self.psi = 0  # 初始角度

    def update_position(self, dt=0.02):  # dt in seconds (20ms)
        """更新卫星位置"""
        delta_psi = self.velocity * dt / (self.Ra + self.h)
        self.psi += delta_psi
        return self.psi

    def get_distance_to_ground(self):
        """计算卫星到地面站距离 - 公式(3)"""
        phi = self.get_elevation_angle()
        # 修正公式以避免负数开方
        distance = math.sqrt(
            self.Ra ** 2 * math.sin(phi) ** 2 + self.h ** 2 +
            2 * self.Ra * self.h * math.cos(phi)
        )
        return distance

    def get_elevation_angle(self):
        """计算仰角 - 公式(1)，添加数值稳定性"""
        cos_psi = math.cos(self.psi)

        # 避免除零错误：当cos(psi)接近±1时使用边界值
        denominator = math.sqrt(max(1e-10, 1 - cos_psi ** 2))  # 添加max确保非负
        if abs(denominator) < 1e-10:  # 数值稳定性阈值
            if cos_psi > 0:
                phi = math.pi / 2  # 90度仰角
            else:
                phi = 0  # 0度仰角
        else:
            numerator = cos_psi - 1 / self.lambda_param
            # 确保atan参数在有效范围内
            phi = math.atan(max(-1e10, min(1e10, numerator / denominator)))

        # 确保仰角在合理范围内 [0, π/2]
        phi = max(0, min(math.pi / 2, abs(phi)))
        return phi

    def get_doppler_shift(self, frequency=17.5e9):
        """计算多普勒频移"""
        theta = math.acos(
            math.sin(self.psi) /
            math.sqrt(1 + self.lambda_param ** 2 - 2 * self.lambda_param * math.cos(self.psi))
        )
        fd = frequency / 3e8 * self.velocity * 1000 * math.cos(theta)  # Hz
        return fd


class ChannelModel:
    """信道模型 - 实现论文中的物理信道模型"""

    def __init__(self, frequency=17.5e9):
        self.frequency = frequency

    def free_space_loss(self, distance_km):
        """自由空间路径损耗"""
        return 32.45 + 20 * math.log10(self.frequency / 1e9) + 20 * math.log10(distance_km)

    def shadowed_rician_fading(self, ms=8, bs=0.129, omega_s=0.372, size=None):
        """阴影瑞利衰落 - 论文公式(4)的简化实现"""
        # 简化实现：使用指数-伽马分布近似
        if size is None:
            return np.random.exponential(1 / bs) * np.random.gamma(ms, omega_s / ms)
        else:
            return np.random.exponential(1 / bs, size) * np.random.gamma(ms, omega_s / ms, size)

    def nakagami_fading(self, m=3, omega=1, size=None):
        """Nakagami-m衰落 - 论文公式(6)"""
        if size is None:
            return np.random.gamma(m, omega / m)
        else:
            return np.random.gamma(m, omega / m, size)

    def get_sat_to_ground_channel(self, distance_km, is_ntn=True, size=None):
        """卫星到地面信道增益"""
        # 路径损耗
        path_loss_db = self.free_space_loss(distance_km)
        path_loss_linear = 10 ** (-path_loss_db / 10)

        # 阴影衰落
        if is_ntn:
            fading = self.shadowed_rician_fading(8, 0.129, 0.372, size=size)
        else:
            fading = self.shadowed_rician_fading(5, 0.251, 0.279, size=size)

        if size is None:
            return path_loss_linear * fading
        else:
            return path_loss_linear * fading

    def get_terrestrial_channel(self, distance_km, path_loss_exp=3.5):
        """地面信道增益"""
        # 路径损耗
        if isinstance(distance_km, np.ndarray):
            path_loss_linear = (1 / distance_km) ** path_loss_exp
            fading = self.nakagami_fading(3, 1, size=distance_km.shape)
        else:
            path_loss_linear = (1 / distance_km) ** path_loss_exp
            fading = self.nakagami_fading(3, 1)

        return path_loss_linear * fading


class SatTerrestrialEnvironment:
    """星地融合频谱共享环境 (缓存优化 + 关键修正版本)"""

    def __init__(self, config):
        self.config = config
        self.satellite = SatelliteOrbitModel()
        self.channel_model = ChannelModel()

        # 网络拓扑参数
        self.num_gnbs = config['num_gnbs']  # M
        self.num_channels = config['num_channels']  # N
        self.users_per_gnb = config['users_per_gnb']  # K
        self.total_bandwidth = 200e6
        self.channel_bandwidth = self.total_bandwidth / self.num_channels

        # QoS要求
        self.terrestrial_qos_threshold = config.get('terrestrial_qos_mbps', 2)
        self.satellite_qos_threshold = config.get('satellite_qos_mbps', 5)

        # 功率常量
        self.power_min = 10 ** (15 / 10) / 1000  # W
        self.power_max = 10 ** (45 / 10) / 1000  # W
        self.sat_power = 1.0  # W
        self.noise_power = 1e-12  # W

        # 初始化位置 - 现在是NumPy数组
        self.ntn_position = np.array([0, 0])
        self.gnb_positions = self._generate_gnb_positions()  # Shape: (M, 2)
        self.user_positions = self._generate_user_positions()  # Shape: (M, K, 2)

        # 卫星信道占用
        self.satellite_channel_allocation = np.random.choice(
            [0, 1], size=self.num_channels, p=[0.6, 0.4]
        )

        # 预计算距离，避免重复计算
        self._precompute_distances()

        # 【新增】预计算并缓存本轮episode的静态信道增益
        self._cache_static_channel_gains()

    def _precompute_distances(self):
        """预计算所有地面链路的距离矩阵"""
        self.distances_gnb_to_user = np.linalg.norm(
            self.gnb_positions[:, np.newaxis, :] - self.user_positions, axis=2
        )
        self.distances_inter_gnb = np.linalg.norm(
            self.user_positions[:, :, np.newaxis, :] - self.gnb_positions[np.newaxis, np.newaxis, :, :],
            axis=3
        )

    def _generate_gnb_positions(self):
        """生成gNB位置 (矢量化)"""
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
        """为每个gNB随机生成用户位置 (矢量化)"""
        num_users_total = self.num_gnbs * self.users_per_gnb
        angles = np.random.uniform(0, 2 * np.pi, num_users_total)
        radii = np.random.uniform(0.5, 3, num_users_total)

        offsets = radii[:, np.newaxis] * np.vstack([np.cos(angles), np.sin(angles)]).T
        offsets = offsets.reshape(self.num_gnbs, self.users_per_gnb, 2)

        return self.gnb_positions[:, np.newaxis, :] + offsets

    def _cache_static_channel_gains(self):
        """在episode开始时计算并缓存静态的地面信道增益"""
        self.cached_gains_gnb_to_users = self.channel_model.get_terrestrial_channel(self.distances_gnb_to_user)

        gains_gnb_interference = self.channel_model.get_terrestrial_channel(self.distances_inter_gnb)

        # 将自己对自己的干扰设置为0 (保持修正)
        for k in range(self.users_per_gnb):
            np.fill_diagonal(gains_gnb_interference[:, k, :], 0)

        self.cached_gains_gnb_interference = gains_gnb_interference

    def get_channel_gains(self):
        """计算所有信道增益矩阵H(t) (使用缓存)"""
        sat_distance = self.satellite.get_distance_to_ground()
        gains_sat_to_ntn = self.channel_model.get_sat_to_ground_channel(sat_distance, size=1, is_ntn=True)
        gains_sat_to_users = self.channel_model.get_sat_to_ground_channel(
            sat_distance, size=(self.num_gnbs, self.users_per_gnb), is_ntn=False
        )

        gains_gnb_to_users = self.cached_gains_gnb_to_users
        gains_gnb_interference = self.cached_gains_gnb_interference

        return gains_sat_to_ntn, gains_sat_to_users, gains_gnb_to_users, gains_gnb_interference

    def calculate_sinr_and_rates(self, channel_gains_tuple, actions, powers):
        """计算SINR和速率 (完全矢量化 + 修正广播错误)"""
        gains_sat_to_ntn, gains_sat_to_users, gains_gnb_to_users, gains_gnb_interference = channel_gains_tuple
        actions_arr = np.array(list(actions.values()))
        powers_arr = np.array(list(powers.values()))

        # --- 1. 卫星用户速率计算 ---
        sat_signal = self.sat_power * gains_sat_to_ntn
        is_interfering_on_sat_ch = self.satellite_channel_allocation[actions_arr] == 1
        sat_interference = np.sum(powers_arr[is_interfering_on_sat_ch] * gains_sat_to_users[is_interfering_on_sat_ch])
        sat_sinr = sat_signal / (sat_interference + self.noise_power)
        satellite_rate = self.channel_bandwidth * np.log2(1 + sat_sinr) / 1e6

        # --- 2. 地面用户速率计算 ---
        signal_power = powers_arr * gains_gnb_to_users
        sat_interference_on_users = self.sat_power * gains_sat_to_users * is_interfering_on_sat_ch

        channel_mask = actions_arr[..., np.newaxis] == np.arange(self.num_channels)
        power_on_channel = powers_arr[..., np.newaxis] * channel_mask

        # --- 【广播错误修正】 ---
        # 1. 计算每个干扰gNB在每个信道上的总功率
        # power_sum_per_gnb_ch shape: (M_interferer, N)
        power_sum_per_gnb_ch = np.sum(power_on_channel, axis=1)

        # 2. 为每个受害用户，找到其使用的信道
        # victim_channels shape: (M_victim, K_victim)
        victim_channels = actions_arr

        # 3. 使用高级索引，为每个受害用户提取所有潜在干扰gNB在对应信道上的功率
        # interfering_powers shape: (M_interferer, M_victim, K_victim)
        interfering_powers = power_sum_per_gnb_ch[:, victim_channels]

        # 4. 调整维度以匹配信道增益矩阵
        # a) interfering_powers -> (M_victim, K_victim, M_interferer)
        interfering_powers = np.transpose(interfering_powers, (1, 2, 0))

        # 5. 计算干扰功率： 功率 * 增益
        # gains_gnb_interference shape: (M_victim, K_victim, M_interferer)
        interference_products = interfering_powers * gains_gnb_interference

        # 6. 对每个用户，将来自所有干扰gNB的干扰求和
        # sum over M_interferer axis (axis=2) -> shape: (M_victim, K_victim)
        inter_cell_interference = np.sum(interference_products, axis=2)

        # --- NOMA干扰计算 (逻辑保持不变) ---
        intra_cell_power_sum_ch = np.sum(power_on_channel, axis=1, keepdims=True)
        noma_interfering_power = intra_cell_power_sum_ch - power_on_channel
        noma_interference_for_user = np.sum(noma_interfering_power * channel_mask, axis=2)
        noma_interference = 0.1 * noma_interference_for_user * gains_gnb_to_users

        # --- 3. 计算最终SINR和速率 ---
        total_interference = sat_interference_on_users + inter_cell_interference + noma_interference
        terrestrial_sinr = signal_power / (total_interference + self.noise_power)
        terrestrial_rates_arr = self.channel_bandwidth * np.log2(1 + terrestrial_sinr) / 1e6

        # --- 4. 格式化输出 ---
        rates = {'satellite': satellite_rate.item(), 'terrestrial': {}}
        for i in range(self.num_gnbs):
            rates['terrestrial'][i] = terrestrial_rates_arr[i].tolist()
        return rates

    def calculate_spectrum_efficiency(self, rates):
        """计算频谱效率"""
        total_rate = rates['satellite']
        for gnb_rates in rates['terrestrial'].values():
            total_rate += sum(gnb_rates)
        return total_rate / (self.total_bandwidth / 1e6)

    def step(self, actions, powers):
        """环境步进 (调用矢量化函数 + 修正奖励函数)"""
        self.satellite.update_position()
        channel_gains_tuple = self.get_channel_gains()
        rates = self.calculate_sinr_and_rates(channel_gains_tuple, actions, powers)

        spectrum_efficiency = self.calculate_spectrum_efficiency(rates)

        # --- 【修正二: 奖励函数整形】 ---
        # 1. 对主奖励进行缩放
        scaled_reward = spectrum_efficiency / 10.0  # 将奖励尺度缩小10倍

        # 2. 加大QoS惩罚力度
        qos_penalty = 0
        if rates['satellite'] < self.satellite_qos_threshold:
            qos_penalty += 1.0  # 显著加大惩罚

        all_terrestrial_rates = np.array(list(rates['terrestrial'].values()))
        qos_violations = np.sum(all_terrestrial_rates < self.terrestrial_qos_threshold)
        qos_penalty += qos_violations * 0.5  # 每个地面用户违规惩罚0.5

        # 最终奖励
        reward = scaled_reward - qos_penalty

        next_state = self._build_state(channel_gains_tuple, actions, powers, rates)
        return next_state, reward, rates

    def _build_state(self, channel_gains_tuple, actions, powers, rates):
        """构建系统状态"""
        # 将当前时刻环境的所有信息打包成一个Python字典，这个字典就是强化学习中的“状态（State）”。
        gains_sat_to_ntn, gains_sat_to_users, gains_gnb_to_users, gains_gnb_interference = channel_gains_tuple
        state = {
            'satellite_position': self.satellite.psi,
            'satellite_channels': self.satellite_channel_allocation,
            'channel_gains': {
                'sat_to_ntn': gains_sat_to_ntn,
                'sat_to_users': gains_sat_to_users,
                'gnb_to_users': gains_gnb_to_users,
                'gnb_interference': gains_gnb_interference
            },
            'prev_actions': actions,
            'prev_powers': powers,
            'prev_rates': rates
        }
        return state

    def get_local_observation(self, gnb_idx, global_state):
        """获取gNB的局部观测 (使用对数缩放以稳定)"""
        # 从全局状态中，为指定的gnb_idx提取与它直接相关的局部信息，例如它自己用户的信道、它受到的卫星干扰信道、卫星的信道占用情况以及它自己上一步的动作。这模拟了现实世界中的部分可观测性。
        gains = global_state['channel_gains']

        local_channels = gains['gnb_to_users'][gnb_idx, :]
        sat_interference_channels = gains['sat_to_users'][gnb_idx, :]

        # 【建议】使用log1p进行缩放，对小的信道增益更稳定
        obs_parts = [
            np.log1p(local_channels / 1e-12),
            np.log1p(sat_interference_channels / 1e-12),
            global_state['satellite_channels']
        ]

        if 'prev_actions' in global_state and gnb_idx in global_state['prev_actions']:
            obs_parts.append(np.array(global_state['prev_actions'][gnb_idx]))
        else:
            obs_parts.append(np.zeros(self.users_per_gnb))

        return np.concatenate(obs_parts).astype(np.float32)


class FELANet(nn.Module):
    """FELA预测网络 - 用于预测未来的网络负载"""

    def __init__(self, input_dim, hidden_dim, output_dim, seq_length):
        super(FELANet, self).__init__()
        self.seq_length = seq_length
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        prediction = self.fc(lstm_out[:, -1, :])
        return prediction


class SatTerrestrialCritic(nn.Module):
    """星地融合网络专用评论家网络"""

    def __init__(self, node_dim, hidden_dim, num_heads=4):
        super(SatTerrestrialCritic, self).__init__()
        self.node_embedding = nn.Linear(node_dim, hidden_dim)
        self.gat1 = GATConv(hidden_dim, hidden_dim, heads=num_heads, dropout=0.1)
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, dropout=0.1)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, node_features, edge_index, actions):
        x = torch.cat([node_features, actions], dim=1)
        x = self.node_embedding(x)
        x = torch.relu(self.gat1(x, edge_index))
        x = torch.relu(self.gat2(x, edge_index))
        global_features = torch.mean(x, dim=0, keepdim=True)
        return self.value_head(global_features)


class SatTerrestrialActor(nn.Module):
    """星地网络双层Actor - 信道分配 + 功率控制"""

    def __init__(self, obs_dim, num_channels, num_users, hidden_dim):
        super(SatTerrestrialActor, self).__init__()
        self.num_channels = num_channels
        self.num_users = num_users

        # 信道分配网络 (DDQN part)
        self.channel_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_users * num_channels),
        )

        # 功率分配网络 (DDPG part)
        self.power_net = nn.Sequential(
            nn.Linear(obs_dim + num_users * num_channels, hidden_dim),  # 修正：obs + channel_assignment维度
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_users),
            nn.Sigmoid()  # 输出0-1，后续缩放到功率范围
        )

    def forward(self, obs, channel_assignment=None):
        # 信道分配
        if channel_assignment is None:
            channel_logits = self.channel_net(obs)
            channel_logits = channel_logits.view(-1, self.num_users, self.num_channels)
            channel_probs = torch.softmax(channel_logits, dim=-1)
        else:
            channel_probs = channel_assignment

        # 功率分配（基于信道分配结果）
        channel_flat = channel_probs.view(channel_probs.size(0), -1)
        power_input = torch.cat([obs, channel_flat], dim=1)
        power_output = self.power_net(power_input)

        return channel_probs, power_output


class SatTerrestrialAgent:
    """星地网络专用智能体"""

    def __init__(self, obs_dim, num_channels, num_users, hidden_dim, lr=1e-4):
        self.obs_dim = obs_dim
        self.num_channels = num_channels
        self.num_users = num_users

        self.actor = SatTerrestrialActor(obs_dim, num_channels, num_users, hidden_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.target_actor = copy.deepcopy(self.actor)

        # 探索参数 - 保持更长时间探索
        self.epsilon = 0.8  # 提高初始探索
        self.epsilon_decay = 0.9995  # 更慢衰减
        self.epsilon_min = 0.05

    def select_action(self, obs, power_min=0.01, power_max=1.0):
        """选择动作：信道分配 + 功率分配"""
        # ε-贪婪探索
        if random.random() < self.epsilon:
            # 随机动作
            channel_actions = np.random.randint(0, self.num_channels, self.num_users)
            power_actions = np.random.uniform(power_min, power_max, self.num_users)
            return channel_actions, power_actions

        # 使用torch.no_grad()的兼容写法
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        channel_probs, power_output = self.actor(obs_tensor)

        # 信道分配（采样）
        channel_dist = torch.distributions.Categorical(channel_probs[0])
        channel_actions = channel_dist.sample().numpy()

        # 功率分配（缩放到实际范围）
        power_actions = power_output[0].detach().numpy() * (power_max - power_min) + power_min

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return channel_actions, power_actions

    def soft_update(self, tau=0.01):
        """软更新目标网络"""
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


class SatTerrestrialHPGAMARLFramework:
    """星地融合HPGA-MARL框架 (修正为真正的集中式Critic)"""

    def __init__(self, config):
        self.config = config
        self.env = SatTerrestrialEnvironment(config)

        # 确定局部和全局观测维度
        self.local_obs_dim = (self.env.users_per_gnb * 2 +  # 本地CSI + 卫星干扰CSI
                              self.env.num_channels +  # 卫星信道占用
                              self.env.users_per_gnb)  # 前一时刻动作

        self.global_obs_dim = self.local_obs_dim * self.env.num_gnbs

        # 创建多个gNB智能体 (Actor)
        self.agents = []
        for _ in range(config['num_gnbs']):
            agent = SatTerrestrialAgent(
                self.local_obs_dim,
                self.env.num_channels,
                self.env.users_per_gnb,
                config['actor_hidden_dim'],
                config['actor_lr']
            )
            self.agents.append(agent)

        # 【修正】创建单一的、输入为全局状态的集中式评论家 (Critic)
        self.global_critic = nn.Sequential(
            nn.Linear(self.global_obs_dim, config['critic_hidden_dim']),
            nn.ReLU(),
            nn.Linear(config['critic_hidden_dim'], config['critic_hidden_dim']),
            nn.ReLU(),
            nn.Linear(config['critic_hidden_dim'], 1)
        )
        self.critic_optimizer = optim.Adam(self.global_critic.parameters(), lr=config['critic_lr'])
        self.target_critic = copy.deepcopy(self.global_critic)

        # 经验回放
        self.replay_buffer = ReplayBuffer(config['buffer_size'])
        self.gamma = config['gamma']
        self.batch_size = config['batch_size']

    def hierarchical_decision(self, global_state):
        """分层决策 (此部分逻辑不变)"""
        actions = {}
        powers = {}
        for gnb_idx, agent in enumerate(self.agents):
            local_obs = self.env.get_local_observation(gnb_idx, global_state)
            channel_actions, power_actions = agent.select_action(
                local_obs, self.env.power_min, self.env.power_max
            )
            actions[gnb_idx] = channel_actions
            powers[gnb_idx] = power_actions
        return actions, powers

    def train_step(self):
        """【修正】使用真正的集中式Critic进行训练"""
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # --- 1. 准备全局状态和下一全局状态的张量 ---
        all_current_obs = []
        all_next_obs = []
        for i in range(len(states)):
            current_global_obs = np.concatenate(
                [self.env.get_local_observation(g, states[i]) for g in range(self.env.num_gnbs)])
            next_global_obs = np.concatenate(
                [self.env.get_local_observation(g, next_states[i]) for g in range(self.env.num_gnbs)])
            all_current_obs.append(current_global_obs)
            all_next_obs.append(next_global_obs)

        global_obs_tensor = torch.FloatTensor(np.array(all_current_obs))
        next_global_obs_tensor = torch.FloatTensor(np.array(all_next_obs))
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1)
        dones_tensor = torch.FloatTensor(dones).unsqueeze(1)

        # --- 2. 训练Critic网络 ---
        with torch.no_grad():
            next_values = self.target_critic(next_global_obs_tensor)
            target_values = rewards_tensor + self.gamma * next_values * (1 - dones_tensor)

        current_values = self.global_critic(global_obs_tensor)
        critic_loss = nn.MSELoss()(current_values, target_values)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.global_critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # --- 3. 训练所有Agent的Actor网络 ---
        with torch.no_grad():
            advantage = target_values - current_values

        for gnb_idx, agent in enumerate(self.agents):
            # 提取该agent的局部观测和动作
            local_obs_list = [self.env.get_local_observation(gnb_idx, s) for s in states]
            local_obs_tensor = torch.FloatTensor(np.array(local_obs_list))

            # 【BUG修正】从回放池中正确解析动作
            # actions 是一个元组，每个元素是一个 combined_actions 字典
            action_list = [a[gnb_idx] for a in actions]
            channel_action_tensor = torch.LongTensor(np.array([ch_pow_pair[0] for ch_pow_pair in action_list]))

            # 计算当前策略下的动作概率和损失
            channel_probs, power_output = agent.actor(local_obs_tensor)

            log_probs = torch.log(channel_probs + 1e-8)
            action_log_probs = log_probs.gather(2, channel_action_tensor.unsqueeze(-1)).squeeze(-1)

            # 损失函数 = -优势 * log(概率)。对一个gNB的所有用户求平均
            actor_loss = - (advantage * action_log_probs.mean(axis=1, keepdim=True)).mean()

            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 1.0)
            agent.actor_optimizer.step()

        # --- 4. 软更新目标网络 ---
        self.soft_update_critic(tau=0.005)
        for agent in self.agents:
            agent.soft_update(tau=0.005)

    def soft_update_critic(self, tau=0.01):
        """为Critic添加软更新"""
        for target_param, param in zip(self.target_critic.parameters(), self.global_critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def reset_environment(self):
        """重置环境 (此部分逻辑不变)"""
        self.env = SatTerrestrialEnvironment(self.config)
        initial_gains = self.env.get_channel_gains()
        initial_actions = {gnb_idx: np.zeros(self.env.users_per_gnb, dtype=int) for gnb_idx in
                           range(self.config['num_gnbs'])}
        initial_powers = {gnb_idx: np.ones(self.env.users_per_gnb) * self.env.power_min for gnb_idx in
                          range(self.config['num_gnbs'])}
        initial_rates = {
            'terrestrial': {gnb_idx: [0] * self.env.users_per_gnb for gnb_idx in range(self.config['num_gnbs'])},
            'satellite': 0}
        return self.env._build_state(initial_gains, initial_actions, initial_powers, initial_rates)


class ReplayBuffer:
    """经验回放缓冲区"""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        return zip(*batch)

    def __len__(self):
        return len(self.buffer)



if __name__ == "__main__":
    # 基于论文Table II的配置参数
    config = {
        'num_gnbs': 3,  # M = 3 gNBs
        'users_per_gnb': 2,  # K = 2 users per gNB
        'num_channels': 10,  # N = 10 channels
        'terrestrial_qos_mbps': 10,
        'satellite_qos_mbps': 20,
        'fela_input_dim': 15,  # 全局系统状态维度 (此参数已弃用)
        'fela_hidden_dim': 64,
        'seq_length': 10,
        'critic_hidden_dim': 256,
        'actor_hidden_dim': 128,
        'critic_lr': 1e-4,
        'actor_lr': 3e-4,  # 演员可以学得快一点
        'buffer_size': 50000,
        'gamma': 0.99,
        'batch_size': 256
    }

    # 创建框架实例
    framework = SatTerrestrialHPGAMARLFramework(config)

    print(f"星地融合网络配置:")
    print(f"- gNB数量: {config['num_gnbs']}")
    print(f"- 每个gNB用户数: {config['users_per_gnb']}")
    print(f"- 局部观测维度: {framework.local_obs_dim}")
    print(f"- 全局观测维度: {framework.global_obs_dim}")

    # 训练过程
    training_episodes = 3000
    best_avg_reward = -float('inf')
    reward_history = []

    for episode in range(training_episodes):
        state = framework.reset_environment()
        episode_reward = 0

        for step in range(50):
            actions, powers = framework.hierarchical_decision(state)
            next_state, reward, rates = framework.env.step(actions, powers)
            episode_reward += reward

            # 【BUG修正】将动作合并为单一字典存入回放池
            combined_actions = {gnb_idx: (actions[gnb_idx], powers[gnb_idx]) for gnb_idx in actions}
            framework.replay_buffer.push(state, combined_actions, reward, next_state, False)

            # 每一步都尝试训练
            if len(framework.replay_buffer) > config['batch_size']:
                framework.train_step()

            state = next_state

        avg_reward = episode_reward / 50
        reward_history.append(avg_reward)
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward

        if episode % 100 == 0:
            # 计算最近100个episode的平均奖励以观察趋势
            avg_100 = np.mean(reward_history[-100:])
            print(f"Episode {episode}: "
                  f"Avg Reward(100) = {avg_100:.2f}, "
                  f"Current Avg Reward = {avg_reward:.2f}, "
                  f"Best Avg Reward = {best_avg_reward:.2f}")

    print("星地融合频谱共享训练完成!")



