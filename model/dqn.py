
import torch.optim as optim
import random
import numpy as np
import torch
import torch.nn as nn
import os
import copy
from collections import deque

from env.myenv import SatTerrestrialEnvironment

class DQN_Joint_Actor(nn.Module):
    """
    DQN联合分配Actor网络。
    输出的是每个用户在 (信道 x 功率等级) 联合空间上的Q值。
    """

    def __init__(self, obs_dim, num_channels, num_users, num_power_levels, hidden_dim):
        super(DQN_Joint_Actor, self).__init__()
        self.num_users = num_users
        self.num_joint_actions = num_channels * num_power_levels

        # 共享主干
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU()
        )

        # 为每个用户创建一个独立的决策头（多分枝结构）
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.num_joint_actions)
            ) for _ in range(num_users)
        ])

    def forward(self, obs):
        features = self.body(obs)
        q_values_per_user = [head(features) for head in self.heads]
        # Shape: (batch_size, num_users, num_joint_actions)
        return torch.stack(q_values_per_user, dim=1)

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

class DQN_Agent:
    """
    独立的DQN智能体，用于联合信道和功率分配。
    """

    def __init__(self, obs_dim, num_channels, num_users, num_power_levels, hidden_dim, lr, device):
        self.device = device
        self.num_channels = num_channels
        self.num_users = num_users
        self.num_power_levels = num_power_levels
        self.num_joint_actions = num_channels * num_power_levels

        self.q_network = DQN_Joint_Actor(obs_dim, num_channels, num_users, num_power_levels, hidden_dim).to(device)
        self.target_network = copy.deepcopy(self.q_network).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        self.epsilon = 0.9
        self.epsilon_decay =  0.999
        self.epsilon_min = 0.005

    def select_action(self, obs, power_levels, use_exploration=True):
        if use_exploration and random.random() < self.epsilon:
            # 探索：为每个用户随机选择一个联合动作
            joint_actions = np.random.randint(0, self.num_joint_actions, self.num_users)
        else:
            # 利用：选择Q值最大的联合动作
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                q_values = self.q_network(obs_tensor)  # Shape: (1, num_users, num_joint_actions)
                joint_actions = torch.argmax(q_values.squeeze(0), dim=1).cpu().numpy()

        # 解码联合动作为 (信道, 功率)
        channel_actions = joint_actions // self.num_power_levels
        power_indices = joint_actions % self.num_power_levels
        power_actions = power_levels[power_indices]

        # 更新epsilon
        if use_exploration:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return channel_actions, power_actions, joint_actions

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())


class DQN_Framework:
    """
    用于DQN联合分配算法的训练和决策框架。
    """

    def __init__(self, config):
        self.config = config
        self.model_params = config["model_params"]['dqn']
        self.env = SatTerrestrialEnvironment(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_power_levels = self.model_params['num_power_levels']
        self.power_levels = np.linspace(self.env.power_min, self.env.power_max, self.num_power_levels)

        # obs_dim = (
        #         config['max_users_per_gnb'] * 2 +  # 对应 padded_local_channels 和 padded_sat_interference
        #         config['num_channels'] +  # 对应 satellite_channels
        #         config['max_users_per_gnb'] +  # 对应 padded_prev_actions
        #         config['num_gnbs']  # 对应 prediction
        # )

        obs_dim = (
                config['max_users_per_gnb'] * 4 +  # 对应4个与用户数相关的部分
                config['num_channels'] +  # 对应 satellite_channels
                config['max_gnbs']  # 对应 padded_prediction
        )

        # 打印调试
        print(f"\n--- DEBUG: DQN_Framework CREATED with obs_dim = {obs_dim} ---\n")


        # self.agents = [
        #     DQN_Agent(obs_dim, self.env.num_channels, self.env.users_per_gnb,
        #               self.num_power_levels, self.model_params['actor_hidden_dim'], self.model_params['actor_lr'], self.device)
        #     for _ in range(self.env.num_gnbs)
        # ]

        self.agents = [
            DQN_Agent(obs_dim, self.env.num_channels, self.config['max_users_per_gnb'],
                      self.num_power_levels, self.model_params['dqn_hidden_dim'], self.model_params['dqn_lr'], self.device)
            for _ in range(self.env.num_gnbs)
        ]

        self.replay_buffer = ReplayBuffer(config['buffer_size'])
        self.batch_size = config['batch_size']
        self.gamma = config['gamma']

    def select_actions(self, global_state, use_exploration=True):
        actions = {}
        powers = {}
        joint_actions_dict = {}

        # 【新增】获取当前环境的真实用户数
        current_users_per_gnb = self.env.users_per_gnb # 真实用户数
        current_gnbs = self.env.num_gnbs # 真实基站数

        # for gnb_idx, agent in enumerate(self.agents):
        # 【核心修改】循环只遍历当前激活的智能体数量
        for gnb_idx in range(current_gnbs):
            agent = self.agents[gnb_idx] # 从完整的agent列表中按索引获取
            # 获取填充后的观测
            max_users = self.config['max_users_per_gnb']
            max_gnbs = self.config['max_gnbs']
            local_obs = self.env.get_local_observation(gnb_idx, global_state, max_gnbs, max_users)

            # Agent返回的是长度为 max_users_per_gnb (例如16) 的完整动作
            full_channel_a, full_power_a, full_joint_a = agent.select_action(local_obs, self.power_levels,
                                                                             use_exploration)

            # ==========================================================
            # 【核心修正】进行输出掩码 (Output Masking)
            # ==========================================================
            # 只截取与当前真实用户数相对应的部分

            # 进行输出掩码（截取与当前环境真实用户数对应的部分）
            actions[gnb_idx] = full_channel_a[:current_users_per_gnb]
            powers[gnb_idx] = full_power_a[:current_users_per_gnb]
            joint_actions_dict[gnb_idx] = full_joint_a[:current_users_per_gnb]
            # ==========================================================

        return actions, powers, joint_actions_dict

        # 【修改后的 train_step 函数】
    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, joint_actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        obs_tensors = []
        next_obs_tensors = []
        max_users = self.config['max_users_per_gnb']
        max_gnbs = self.config['max_gnbs']
        for gnb_idx in range(self.env.num_gnbs):
            obs_list = [self.env.get_local_observation(gnb_idx, s, max_gnbs, max_users) for s in states]
            next_obs_list = [self.env.get_local_observation(gnb_idx, s, max_gnbs, max_users) for s in next_states]
            obs_tensors.append(torch.FloatTensor(np.array(obs_list)).to(self.device))
            next_obs_tensors.append(torch.FloatTensor(np.array(next_obs_list)).to(self.device))

        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones_tensor = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # 独立训练每个智能体 (Independent Q-Learning)
        for gnb_idx, agent in enumerate(self.agents):
            # ==========================================================
            # 【核心修改区域：Double DQN 目标Q值计算】
            # ==========================================================
            with torch.no_grad():
                # --- 1. 使用主网络(q_network)选择下一状态的最佳动作 ---
                # main_next_q_values shape: (batch, num_users, num_joint_actions)
                main_next_q_values = agent.q_network(next_obs_tensors[gnb_idx])
                # best_next_actions shape: (batch, num_users)
                best_next_actions = torch.argmax(main_next_q_values, dim=2)

                # --- 2. 使用目标网络(target_network)评估这些最佳动作的Q值 ---
                # target_next_q_values shape: (batch, num_users, num_joint_actions)
                target_next_q_values = agent.target_network(next_obs_tensors[gnb_idx])

                # 使用gather根据best_next_actions选出对应的Q值
                # q_values_of_best_action shape: (batch, num_users)
                q_values_of_best_action = target_next_q_values.gather(2, best_next_actions.unsqueeze(-1)).squeeze(
                    -1)

                # 【说明】这里是与标准DQN最本质的区别。
                # 标准DQN是直接在target_next_q_values上取max，而Double DQN是在target_next_q_values上
                # 根据主网络选出的动作索引(best_next_actions)来取值。

                # 将每个用户的Q值求平均，作为下一状态的价值
                next_state_value = torch.mean(q_values_of_best_action, dim=1, keepdim=True)

                # --- 3. 计算最终的目标Q值 ---
                target_q = rewards_tensor + self.gamma * next_state_value * (1 - dones_tensor)
            # ==========================================================
            # 【修改结束】
            # ==========================================================

            # --- 计算当前Q值 (Current Q-Value) --- (这部分逻辑不变)
            agent_joint_actions = torch.LongTensor(np.array([a[gnb_idx] for a in joint_actions])).to(self.device)
            current_q_all_actions = agent.q_network(obs_tensors[gnb_idx])
            current_q_per_user = current_q_all_actions.gather(2, agent_joint_actions.unsqueeze(-1)).squeeze(-1)
            current_q = torch.mean(current_q_per_user, dim=1, keepdim=True)

            # --- 计算损失并更新 --- (这部分逻辑不变)
            loss = nn.MSELoss()(current_q, target_q)
            agent.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.q_network.parameters(), 1.0)
            agent.optimizer.step()

    def save_models(self, filename="best_dqn_model.pth"):
        checkpoint = {'actors': [agent.q_network.state_dict() for agent in self.agents]}
        torch.save(checkpoint, filename)

    def load_models(self, filename="best_dqn_model.pth"):
        """加载所有DQN智能体的预训练模型"""
        if not os.path.exists(filename):
            print(f"错误: DQN模型文件未找到于 {filename}。无法运行测试模式。")
            exit()

        checkpoint = torch.load(filename, map_location=self.device)
        # 假设保存的 checkpoint 格式为 {'actors': [state_dict_1, state_dict_2, ...]}
        if 'actors' in checkpoint and len(checkpoint['actors']) == len(self.agents):
            for i, agent in enumerate(self.agents):
                agent.q_network.load_state_dict(checkpoint['actors'][i])
                # 将权重同步到目标网络，以确保一致性
                agent.target_network.load_state_dict(checkpoint['actors'][i])
            print(f"DQN 模型已从 {filename} 加载。")
        else:
            print(f"错误: 模型文件 {filename} 格式不正确或智能体数量不匹配。")
            exit()

