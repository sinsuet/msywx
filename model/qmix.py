# ---------------- QMIX 专用网络结构 ----------------
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

from env.myenv import SatTerrestrialEnvironment


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

# ---------------- QMIX 专用网络结构 ----------------

class QMIX_Agent_Network(nn.Module):
    """
    QMIX中每个独立智能体的网络 (DRQN)。
    它为单个gNB下的所有用户输出并行的Q值。
    """

    def __init__(self, obs_dim, num_channels, num_users, num_power_levels, hidden_dim):
        super(QMIX_Agent_Network, self).__init__()
        self.num_users = num_users
        self.num_joint_actions = num_channels * num_power_levels

        # 特征提取
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        # GRU用于记忆历史信息
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        # 多分支输出头
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, self.num_joint_actions) for _ in range(num_users)
        ])

    def forward(self, obs, hidden_state):
        x = torch.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.gru.hidden_size)
        h_out = self.gru(x, h_in)

        # 为每个用户（分支）计算Q值
        q_values_per_user = [head(h_out) for head in self.heads]
        # 将所有用户的Q值堆叠起来
        q_values = torch.stack(q_values_per_user, dim=1)

        return q_values, h_out


class QMIX_Mixing_Network(nn.Module):
    """
    QMIX的核心：混合网络。
    """

    def __init__(self, num_agents, state_dim, hidden_dim):
        super(QMIX_Mixing_Network, self).__init__()
        self.num_agents = num_agents
        # ==========================================================
        # 【核心修改】将 hidden_dim 保存为类属性
        # ==========================================================
        self.hidden_dim = hidden_dim
        # ==========================================================

        # Hypernetworks: 根据全局状态生成混合网络的权重和偏置
        self.hyper_w1 = nn.Sequential(nn.Linear(state_dim, hidden_dim * num_agents), nn.ReLU())
        self.hyper_b1 = nn.Linear(state_dim, hidden_dim)

        self.hyper_w2 = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU())
        self.hyper_b2 = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, agent_qs, global_state):
        # agent_qs shape: (batch_size, num_agents)
        # global_state shape: (batch_size, state_dim)
        batch_size = agent_qs.size(0)

        agent_qs = agent_qs.view(batch_size, 1, self.num_agents)

        # Layer 1
        w1 = torch.abs(self.hyper_w1(global_state))
        b1 = self.hyper_b1(global_state)
        w1 = w1.view(batch_size, self.num_agents, self.hidden_dim)
        b1 = b1.view(batch_size, 1, self.hidden_dim)

        hidden = torch.relu(torch.bmm(agent_qs, w1) + b1)

        # Layer 2
        w2 = torch.abs(self.hyper_w2(global_state))
        b2 = self.hyper_b2(global_state)
        w2 = w2.view(batch_size, self.hidden_dim, 1)
        b2 = b2.view(batch_size, 1, 1)

        q_tot = torch.bmm(hidden, w2) + b2

        return q_tot.view(batch_size, -1)

class QMIX_Framework:
    """
    【重构版】QMIX算法的完整训练和决策框架。
    将获取局部观测的逻辑封装在框架内部。
    """

    def __init__(self, config):
        self.config = config
        self.env = SatTerrestrialEnvironment(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        qmix_params = config['model_params']['qmix']
        self.num_power_levels = qmix_params['num_power_levels']
        self.power_levels = np.linspace(self.env.power_min, self.env.power_max, self.num_power_levels)
        self.num_joint_actions = self.env.num_channels * self.num_power_levels

        self.local_obs_dim = (
                config['max_users_per_gnb'] * 4 + config['num_channels'] + config['max_gnbs']
        )
        self.global_state_dim = self.local_obs_dim * config['max_gnbs']
        self.num_agents = config['max_gnbs']
        self.hidden_dim = qmix_params['hidden_dim']

        self.agent_networks = [
            QMIX_Agent_Network(self.local_obs_dim, self.env.num_channels, config['max_users_per_gnb'],
                               self.num_power_levels, self.hidden_dim).to(self.device)
            for _ in range(self.num_agents)
        ]
        self.mixing_network = QMIX_Mixing_Network(self.num_agents, self.global_state_dim, self.hidden_dim).to(
            self.device)

        self.target_agent_networks = copy.deepcopy(self.agent_networks)
        self.target_mixing_network = copy.deepcopy(self.mixing_network)

        network_params = list(self.mixing_network.parameters()) + [p for net in self.agent_networks for p in
                                                                   net.parameters()]
        self.optimizer = optim.Adam(network_params, lr=qmix_params['lr'])

        self.replay_buffer = ReplayBuffer(config['buffer_size'])
        self.batch_size = config['batch_size']
        self.gamma = config['gamma']

        self.epsilon = qmix_params['epsilon_start']
        self.epsilon_decay = qmix_params['epsilon_decay']
        self.epsilon_min = qmix_params['epsilon_min']

    def select_actions(self, global_state, hidden_states, use_exploration=True):
        """
        【修改】接收 global_state，并在内部获取局部观测。
        """
        actions, powers, joint_actions_dict = {}, {}, {}
        next_hidden_states = []

        # 【新增】从当前环境中获取真实的基站数和用户数
        current_gnbs = self.env.num_gnbs
        current_users_per_gnb = self.env.users_per_gnb

        # 【修改】封装获取局部观测的逻辑
        local_obs_list = [self.env.get_local_observation(g, global_state, self.config['max_gnbs'], self.config['max_users_per_gnb'])
                          for g in range(current_gnbs)]

        if use_exploration and random.random() < self.epsilon:
            for gnb_idx in range(current_gnbs):
                # 探索时也只为真实用户生成动作
                joint_actions = np.random.randint(0, self.num_joint_actions, current_users_per_gnb)
                actions[gnb_idx] = joint_actions // self.num_power_levels
                powers[gnb_idx] = self.power_levels[joint_actions % self.num_power_levels]
                joint_actions_dict[gnb_idx] = joint_actions
            # 探索步只更新激活的智能体的隐藏状态（或保持不变）
            # 为简单起见，我们保持不变，因为GRU未被有效使用
            next_hidden_states = [hidden_states[i] for i in range(current_gnbs)]
        else:
            with torch.no_grad():
                for gnb_idx in range(current_gnbs):
                    obs_tensor = torch.FloatTensor(local_obs_list[gnb_idx]).unsqueeze(0).to(self.device)
                    # Agent网络返回的是全尺寸的Q值 (max_users_per_gnb, num_joint_actions)
                    q_values, h = self.agent_networks[gnb_idx](obs_tensor, hidden_states[gnb_idx])

                    # 【核心修正】进行输出掩码 (Output Masking)
                    # 1. 只截取与当前真实用户数对应的Q值部分
                    q_values_masked = q_values.squeeze(0)[:current_users_per_gnb, :]

                    # 2. 在掩码后的Q值上选择最佳动作
                    best_joint_actions = torch.argmax(q_values_masked, dim=1).cpu().numpy()

                    # 3. 解码并存储
                    actions[gnb_idx] = best_joint_actions // self.num_power_levels
                    powers[gnb_idx] = self.power_levels[best_joint_actions % self.num_power_levels]
                    joint_actions_dict[gnb_idx] = best_joint_actions
                    next_hidden_states.append(h)

        # # 【新增】从全局状态生成所有智能体的局部观测列表
        # local_obs_list = [self.env.get_local_observation(g, global_state, self.config['max_gnbs'],
        #                                                  self.config['max_users_per_gnb'])
        #                   for g in range(self.env.num_gnbs)]

        # if use_exploration and random.random() < self.epsilon:
        #     for gnb_idx in range(self.env.num_gnbs):
        #         joint_actions = np.random.randint(0, self.num_joint_actions, self.env.users_per_gnb)
        #         actions[gnb_idx] = joint_actions // self.num_power_levels
        #         powers[gnb_idx] = self.power_levels[joint_actions % self.num_power_levels]
        #         joint_actions_dict[gnb_idx] = joint_actions
        #     # 在探索步，隐藏状态不通过网络，保持不变
        #     next_hidden_states = hidden_states
        # else:
        #     with torch.no_grad():
        #         for gnb_idx in range(self.env.num_gnbs):
        #             obs_tensor = torch.FloatTensor(local_obs_list[gnb_idx]).unsqueeze(0).to(self.device)
        #             # 确保为正确的agent传入其对应的隐藏状态
        #             q_values, h = self.agent_networks[gnb_idx](obs_tensor, hidden_states[gnb_idx])
        #
        #             best_joint_actions = torch.argmax(q_values.squeeze(0), dim=1).cpu().numpy()
        #
        #             actions[gnb_idx] = best_joint_actions // self.num_power_levels
        #             powers[gnb_idx] = self.power_levels[best_joint_actions % self.num_power_levels]
        #             joint_actions_dict[gnb_idx] = best_joint_actions
        #             next_hidden_states.append(h)

        if use_exploration:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return actions, powers, joint_actions_dict, next_hidden_states

    def _prepare_batch_tensors(self, states, next_states):
        """
        【新增】一个私有辅助方法，用于从批次数据中准备所有必需的张量。
        """
        all_obs, all_next_obs = [[] for _ in range(self.num_agents)], [[] for _ in range(self.num_agents)]
        for s in states:
            for i in range(self.num_agents): all_obs[i].append(
                self.env.get_local_observation(i, s, self.config['max_gnbs'], self.config['max_users_per_gnb']))
        for s_next in next_states:
            for i in range(self.num_agents): all_next_obs[i].append(
                self.env.get_local_observation(i, s_next, self.config['max_gnbs'],
                                               self.config['max_users_per_gnb']))

        obs_tensors = [torch.FloatTensor(np.array(o)).to(self.device) for o in all_obs]
        next_obs_tensors = [torch.FloatTensor(np.array(o)).to(self.device) for o in all_next_obs]

        global_state_tensor = torch.cat(obs_tensors, dim=1)
        next_global_state_tensor = torch.cat(next_obs_tensors, dim=1)

        return obs_tensors, next_obs_tensors, global_state_tensor, next_global_state_tensor

    def train_step(self):
        """
        【修改】使用 _prepare_batch_tensors 辅助方法，使代码更整洁。
        """
        if len(self.replay_buffer) < self.batch_size:
            return

        states, joint_actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones_tensor = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # 【修改】调用辅助方法来准备所有观测张量
        obs_tensors, next_obs_tensors, global_state_tensor, next_global_state_tensor = self._prepare_batch_tensors(
            states, next_states)

        with torch.no_grad():
            target_agent_qs = []

            # 初始化一个零向量的隐藏状态，用于批次训练
            batch_hidden_state = torch.zeros(self.batch_size, self.hidden_dim).to(self.device)

            for i in range(self.num_agents):
                main_next_q_vals, _ = self.agent_networks[i](next_obs_tensors[i], batch_hidden_state)
                best_next_actions_per_user = torch.argmax(main_next_q_vals, dim=2)

                target_next_q_vals, _ = self.target_agent_networks[i](next_obs_tensors[i], batch_hidden_state)
                q_val_per_user = target_next_q_vals.gather(2, best_next_actions_per_user.unsqueeze(-1)).squeeze(-1)

                target_agent_qs.append(torch.mean(q_val_per_user, dim=1))

            target_agent_qs_tensor = torch.stack(target_agent_qs, dim=1)
            target_q_tot = self.target_mixing_network(target_agent_qs_tensor, next_global_state_tensor)

            y = rewards_tensor + self.gamma * target_q_tot * (1 - dones_tensor)

        current_agent_qs = []
        batch_hidden_state = torch.zeros(self.batch_size, self.hidden_dim).to(self.device)
        for i in range(self.num_agents):
            q_vals, _ = self.agent_networks[i](obs_tensors[i], batch_hidden_state)

            agent_actions = torch.LongTensor(np.array([a[i] for a in joint_actions])).to(self.device)
            q_val_per_user = q_vals.gather(2, agent_actions.unsqueeze(-1)).squeeze(-1)

            current_agent_qs.append(torch.mean(q_val_per_user, dim=1))

        current_agent_qs_tensor = torch.stack(current_agent_qs, dim=1)
        current_q_tot = self.mixing_network(current_agent_qs_tensor, global_state_tensor)

        loss = nn.MSELoss()(current_q_tot, y.detach())
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in self.optimizer.param_groups[0]['params'] if p.grad is not None],
                                       1.0)
        self.optimizer.step()

    def update_target_networks(self, tau):
        # 软更新Agent网络
        for i in range(self.num_agents):
            for target_param, param in zip(self.target_agent_networks[i].parameters(),
                                           self.agent_networks[i].parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        # 软更新Mixing网络
        for target_param, param in zip(self.target_mixing_network.parameters(), self.mixing_network.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save_models(self, filename):
        checkpoint = {
            'agent_networks': [net.state_dict() for net in self.agent_networks],
            'mixing_network': self.mixing_network.state_dict()
        }
        torch.save(checkpoint, filename)

    def load_models(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        for i in range(self.num_agents):
            self.agent_networks[i].load_state_dict(checkpoint['agent_networks'][i])
            self.target_agent_networks[i].load_state_dict(checkpoint['agent_networks'][i])
        self.mixing_network.load_state_dict(checkpoint['mixing_network'])
        self.target_mixing_network.load_state_dict(checkpoint['mixing_network'])
        print(f"QMIX模型已从 {filename} 加载。")


