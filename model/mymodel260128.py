# -*- coding: utf-8 -*-
# 最终版本: 包含PRB利用率预测、训练/测试模式切换、模型保存/加载、性能记录
# 修改点：引入预测辅助任务 (Predictive Auxiliary Task)

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
    """(此类无需修改)"""

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


# ==============================================================================
# 新增：带有辅助任务头的 Critic 网络
# ==============================================================================
class SatTerrestrialCritic(nn.Module):
    """
    带有预测辅助任务的Critic网络
    主任务: 输出 Q 值 (1维)
    辅助任务: 预测下一时刻的环境动态 (如PRB利用率/干扰强度)
    """

    def __init__(self, global_obs_dim, hidden_dim, aux_output_dim):
        super(SatTerrestrialCritic, self).__init__()

        # 共享特征提取层
        self.base = nn.Sequential(
            nn.Linear(global_obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )

        # 主任务头: Q值估计
        self.q_head = nn.Linear(hidden_dim, 1)

        # 辅助任务头: 环境动态预测
        # aux_output_dim 对应预测向量的长度 (这里是 max_gnbs)
        self.aux_head = nn.Linear(hidden_dim, aux_output_dim)

    def forward(self, global_obs):
        features = self.base(global_obs)
        q_value = self.q_head(features)
        aux_pred = self.aux_head(features)
        return q_value, aux_pred


class MAAC_Critic(nn.Module):
    """
    基于注意力机制的 Critic 网络 (解决维度爆炸与动态拓扑问题)
    """

    def __init__(self, sa_dim, hidden_dim, num_heads=4, aux_output_dim=None):
        """
        sa_dim: 单个智能体的 (Obs_dim + Act_dim) 的维度。注意是"单个"，不再是全局拼接的维度！
        """
        super(MAAC_Critic, self).__init__()

        # 1. 特征编码器 (Encoder)
        # 作用：把每个智能体原本长短不一或原始的特征，映射到统一的 hidden_dim 空间
        self.encoder = nn.Sequential(
            nn.Linear(sa_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 2. 多头注意力层 (Multi-Head Attention)
        # 作用：处理 N 个智能体之间的交互
        # batch_first=True -> 输入形状 (Batch, Seq_Len, Dim)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

        # 3. Q值输出头 (Q-Head)
        # 输入：自己的特征 + 注意力聚合后的全局特征 (2 * hidden_dim)
        self.q_net = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # 4. (保留你的上一个创新) 辅助任务预测头
        if aux_output_dim:
            self.aux_net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),  # 这里只利用自身的特征预测环境动态，或者也可以用聚合特征
                nn.ReLU(),
                nn.Linear(hidden_dim, aux_output_dim)
            )
        self.aux_output_dim = aux_output_dim

    def forward(self, batch_obs, batch_act):
        """
        batch_obs: (Batch, N_Agents, Obs_Dim)
        batch_act: (Batch, N_Agents, Act_Dim)
        """
        batch_size = batch_obs.shape[0]
        n_agents = batch_obs.shape[1]

        # 1. 拼接状态和动作: (Batch, N, Obs+Act)
        inputs = torch.cat([batch_obs, batch_act], dim=2)

        # 2. 独立编码: (Batch, N, Hidden)
        embeddings = self.encoder(inputs)

        # 3. 注意力计算
        # query=embeddings, key=embeddings, value=embeddings
        # attn_output: (Batch, N, Hidden) -> 每个智能体收到的加权全局信息
        # attn_weights: (Batch, N, N) -> 干扰图谱 (可用于论文画图!)
        attn_output, attn_weights = self.attention(embeddings, embeddings, embeddings)

        # 4. 融合特征: 自己的特征 + 全局上下文
        # (Batch, N, 2*Hidden)
        combined_features = torch.cat([embeddings, attn_output], dim=2)

        # 5. 输出 Q 值
        q_values = self.q_net(combined_features)  # (Batch, N, 1)

        # 6. 输出辅助预测 (可选)
        aux_preds = None
        if self.aux_output_dim:
            # 这里可以选择用 embeddings (只看自己) 或 attn_output (看全局)
            # 假设辅助任务是预测全局干扰，用 attn_output 更好
            aux_preds = self.aux_net(attn_output)

        return q_values.squeeze(-1), aux_preds, attn_weights

class ReplayBuffer:
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


class SatTerrestrialHPGAMARLFramework:
    """星地融合HPGA-MARL框架 (集成预测辅助任务)"""

    def __init__(self, config):
        self.config = config
        self.model_name = config['model_name']
        self.model_params = config['model_params'][self.model_name]
        self.env = SatTerrestrialEnvironment(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 辅助任务权重 (可以放入config中调节)
        self.beta_aux = 0.5

        # 【核心修正】正确计算观测维度，与 myenv.py 对齐
        # myenv.py 输出包含:
        # 1. log1p(local_ch) [Max_U]
        # 2. log1p(cong_adj_ch) [Max_U]
        # 3. log1p(sat_int) [Max_U]
        # 4. sat_ch [Num_CH]
        # 5. prev_act [Max_U]
        # 6. prediction [Max_GNB]
        # 总计: 4 * Max_U + Num_CH + Max_GNB

        max_users = config['max_users_per_gnb']
        max_gnbs = config['max_num_gnbs']
        num_channels = self.env.num_channels

        self.local_obs_dim = (
                4 * max_users +
                num_channels +
                max_gnbs
        )

        self.global_obs_dim = self.local_obs_dim * self.env.num_gnbs

        # 创建智能体
        self.agents = []
        for _ in range(config['num_gnbs']):
            agent = SatTerrestrialAgent(
                self.local_obs_dim,
                self.env.num_channels,
                self.config['max_users_per_gnb'],
                self.model_params['actor_hidden_dim'],
                self.model_params['actor_lr'],
                device=self.device
            )
            self.agents.append(agent)

        # 【修改】使用带有辅助头的 Critic 类
        # 辅助任务的目标维度是 max_gnbs (预测每个基站的负载/拥塞)
        self.global_critic = SatTerrestrialCritic(
            self.global_obs_dim,
            self.model_params['critic_hidden_dim'],
            aux_output_dim=max_gnbs
        )

        # 将网络移动到指定设备
        for agent in self.agents:
            agent.actor.to(self.device)
            agent.target_actor.to(self.device)
        self.global_critic.to(self.device)

        self.critic_optimizer = optim.Adam(self.global_critic.parameters(), lr=self.model_params['critic_lr'])
        self.target_critic = copy.deepcopy(self.global_critic).to(self.device)

        self.replay_buffer = ReplayBuffer(config['buffer_size'])
        self.gamma = config['gamma']
        self.batch_size = config['batch_size']

    def save_models(self, filename="best_model.pth"):
        checkpoint = {
            'global_critic': self.global_critic.state_dict(),
            'actors': [agent.actor.state_dict() for agent in self.agents]
        }
        torch.save(checkpoint, filename)

    def load_models(self, filename="best_model.pth"):
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
        actions = {}
        powers = {}
        current_users_per_gnb = self.env.users_per_gnb
        current_gnbs = self.env.num_gnbs

        for gnb_idx in range(current_gnbs):
            agent = self.agents[gnb_idx]
            local_obs = self.env.get_local_observation(
                gnb_idx, global_state, self.config['max_num_gnbs'], self.config['max_users_per_gnb']
            )

            full_channel_actions, full_power_actions = agent.select_action(
                local_obs, self.env.power_min, self.env.power_max, use_exploration=use_exploration
            )

            actions[gnb_idx] = full_channel_actions[:current_users_per_gnb]
            powers[gnb_idx] = full_power_actions[:current_users_per_gnb]

        return actions, powers

    def train_step(self):
        """训练步骤 (集成预测辅助 Loss)"""
        if len(self.replay_buffer) < self.batch_size:
            return

        sampled_data = self.replay_buffer.sample(self.batch_size)
        if sampled_data is None: return
        states, actions, rewards, next_states, dones = sampled_data

        # --- 1. 准备数据 ---
        cached_current_local_obs = [[] for _ in range(self.env.num_gnbs)]
        cached_next_local_obs = [[] for _ in range(self.env.num_gnbs)]
        max_users = self.config['max_users_per_gnb']
        max_gnbs = self.config['max_num_gnbs']

        for i in range(len(states)):
            for gnb_idx in range(self.env.num_gnbs):
                cached_current_local_obs[gnb_idx].append(
                    self.env.get_local_observation(gnb_idx, states[i], max_gnbs, max_users)
                )
                cached_next_local_obs[gnb_idx].append(
                    self.env.get_local_observation(gnb_idx, next_states[i], max_gnbs, max_users)
                )

        cached_current_local_obs_tensors = [torch.FloatTensor(np.array(obs)).to(self.device) for obs in
                                            cached_current_local_obs]
        cached_next_local_obs_tensors = [torch.FloatTensor(np.array(obs)).to(self.device) for obs in
                                         cached_next_local_obs]

        global_obs_tensor = torch.cat(cached_current_local_obs_tensors, dim=1)
        next_global_obs_tensor = torch.cat(cached_next_local_obs_tensors, dim=1)

        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones_tensor = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # --- 2. 提取辅助任务的目标 (真实/预测的环境动态) ---
        # 目标：预测 Next State 中的 "预测向量" (Environmental Dynamics)
        # 预测向量在 Local Obs 的最后 max_gnbs 位
        # 由于 Global Obs 是所有 Local Obs 的拼接，且所有 Agent 看到的 Global Prediction 是一样的
        # 我们直接取第一个 Agent 的 Local Obs 的最后部分作为目标

        # next_global_obs_tensor 的结构: [Agent1_Obs, Agent2_Obs, ...]
        # Agent1_Obs 的长度是 self.local_obs_dim
        # Prediction 位于 Agent1_Obs 的最后 max_gnbs 位
        target_prediction = next_global_obs_tensor[:, self.local_obs_dim - max_gnbs: self.local_obs_dim]

        # --- 3. Critic 训练 (含辅助任务) ---
        with torch.no_grad():
            # Target Critic 只输出 Q 值 (Target Q)，不需要计算辅助预测
            next_q_values, _ = self.target_critic(next_global_obs_tensor)
            target_q_values = rewards_tensor + self.gamma * next_q_values * (1 - dones_tensor)

        # Current Critic 输出 Q 值 和 辅助预测
        current_q_values, current_aux_pred = self.global_critic(global_obs_tensor)

        # 计算 Loss
        td_loss = nn.MSELoss()(current_q_values, target_q_values)
        aux_loss = nn.MSELoss()(current_aux_pred, target_prediction)

        # 总 Loss = TD Loss + beta * Aux Loss
        total_critic_loss = td_loss + self.beta_aux * aux_loss

        self.critic_optimizer.zero_grad()
        total_critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.global_critic.parameters(), 1.0)
        self.critic_optimizer.step()

        with torch.no_grad():
            advantage = target_q_values - current_q_values

        # --- 4. Actor 训练 ---
        for gnb_idx, agent in enumerate(self.agents):
            local_obs_tensor = cached_current_local_obs_tensors[gnb_idx]

            action_list = [a[gnb_idx] for a in actions]
            channel_action_tensor = torch.LongTensor(np.array([ch_pow_pair[0] for ch_pow_pair in action_list])).to(
                self.device)

            channel_probs, power_output = agent.actor(local_obs_tensor)
            log_probs = torch.log(channel_probs + 1e-8)
            action_log_probs = log_probs.gather(2, channel_action_tensor.unsqueeze(-1)).squeeze(-1)
            actor_loss = - (advantage.detach() * action_log_probs.mean(axis=1, keepdim=True)).mean()

            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 1.0)
            agent.actor_optimizer.step()

        # --- 5. 软更新 ---
        self.soft_update_critic(tau=0.005)
        for agent in self.agents:
            agent.soft_update(tau=0.005)

    def soft_update_critic(self, tau=0.01):
        for target_param, param in zip(self.target_critic.parameters(), self.global_critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def reset_environment(self):
        # 保持原有逻辑不变
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