# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import copy
import os

# 【核心修正】导入环境类，确保可以与环境交互
from env.myenv2 import SatTerrestrialEnvironment


class PPOMemory:
    """
    PPO 专用存储器，用于存储一个 Episode 或一定步数的轨迹。
    用于计算 GAE (Generalized Advantage Estimation)。
    """

    def __init__(self, batch_size, gamma=0.99, gae_lambda=0.95, device='cpu'):
        self.states = []
        self.obs = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.costs = []  # 新增：用于存储约束代价 (QoS/拥塞)
        self.dones = []
        self.values = []
        self.aux_targets = []  # 新增：用于存储辅助任务的真实标签

        self.batch_size = batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device

    def store(self, state, obs, action, prob, val, reward, cost, done, aux_target):
        self.states.append(state)
        self.obs.append(obs)
        self.actions.append(action)
        self.probs.append(prob)
        self.values.append(val)
        self.rewards.append(reward)
        self.costs.append(cost)
        self.dones.append(done)
        self.aux_targets.append(aux_target)

    def clear(self):
        self.states = []
        self.obs = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.costs = []
        self.dones = []
        self.values = []
        self.aux_targets = []

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]

        # --- 【核心修正】正确解压并转换 actions ---
        # self.actions 结构是 [(ch_t0, pow_t0), (ch_t1, pow_t1), ...]
        # 我们将其拆分为两个独立的 (N_steps, Num_Agents, ...) 的数组
        ch_actions_list = [a[0] for a in self.actions]
        pow_actions_list = [a[1] for a in self.actions]

        all_ch_actions = np.array(ch_actions_list)
        all_pow_actions = np.array(pow_actions_list)
        # ----------------------------------------

        return np.array(self.states), \
            np.array(self.obs), \
            all_ch_actions, \
            all_pow_actions, \
            np.array(self.probs), \
            np.array(self.values), \
            np.array(self.rewards), \
            np.array(self.costs), \
            np.array(self.dones), \
            np.array(self.aux_targets), \
            batches

    def compute_gae(self, next_value, current_lambda, external_rewards=None):
        """
        允许传入外部缩放过的 rewards
        """
        # 如果传入了 external_rewards 就用它，否则用存的 self.rewards
        rewards = external_rewards if external_rewards is not None else np.array(self.rewards)
        costs = np.array(self.costs)
        dones = np.array(self.dones)
        values = np.array(self.values)

        # 混合奖励: Scaled_Reward - Lambda * Cost
        # 注意：Cost 本身已经是 0~1 之间了，Lambda 也就几，这个量级(个位数)是匹配的
        combined_rewards = rewards - current_lambda * costs

        advantages = np.zeros_like(combined_rewards)
        lastgaelam = 0

        for t in reversed(range(len(combined_rewards))):
            if t == len(combined_rewards) - 1:
                nextnonterminal = 1.0 - dones[t]
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t]
                nextvalues = values[t + 1]

            # GAE 公式
            delta = combined_rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam

        returns = advantages + values
        return returns, advantages


class SelfAttentionLayer(nn.Module):
    """
    基础的 Self-Attention 模块，用于处理多智能体之间的拓扑关系。
    """

    def __init__(self, input_dim, hidden_dim, num_heads=4):
        super(SelfAttentionLayer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x shape: (batch_size, num_agents, input_dim)
        x_embed = self.input_proj(x)
        attn_output, _ = self.multihead_attn(x_embed, x_embed, x_embed)
        return self.layer_norm(x_embed + attn_output)


class AttentionCritic(nn.Module):
    """
    【创新点】基于 Attention 的 Critic 网络。
    """

    def __init__(self, obs_dim, hidden_dim, aux_output_dim):
        super(AttentionCritic, self).__init__()
        self.attention = SelfAttentionLayer(obs_dim, hidden_dim)

        # 价值头
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # 输出标量 V(s)
        )

        # 【创新点】辅助预测头 (Auxiliary Head)
        self.aux_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, aux_output_dim)
        )

    def forward(self, global_obs_seq):
        # global_obs_seq: (Batch, Num_Agents, Obs_Dim)

        # 1. 提取拓扑特征
        features = self.attention(global_obs_seq)  # (Batch, Num_Agents, Hidden)

        # 2. 全局池化 (Mean Pooling)
        global_features = torch.mean(features, dim=1)  # (Batch, Hidden)

        # 3. 输出价值
        value = self.value_head(global_features)

        # 4. 输出辅助预测
        aux_pred = self.aux_head(global_features)

        return value, aux_pred


class MAPPOActor(nn.Module):
    """
    MAPPO Actor 网络。
    """

    def __init__(self, obs_dim, num_channels, num_users, hidden_dim):
        super(MAPPOActor, self).__init__()
        self.num_channels = num_channels
        self.num_users = num_users

        self.shared_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # 信道选择分支
        self.channel_head = nn.Linear(hidden_dim, num_users * num_channels)

        # 功率分配分支
        self.power_mean_head = nn.Sequential(
            nn.Linear(hidden_dim, num_users),
            nn.Sigmoid()
        )
        # 可学习的 LogStd
        self.power_log_std = nn.Parameter(torch.zeros(1, num_users) - 0.5)

    def forward(self, obs):
        x = self.shared_net(obs)

        # 信道 Logits
        channel_logits = self.channel_head(x).view(-1, self.num_users, self.num_channels)

        # 功率 Mean
        power_mean = self.power_mean_head(x)
        power_std = torch.exp(self.power_log_std).expand_as(power_mean)

        return channel_logits, power_mean, power_std

    def evaluate(self, obs, action_channels, action_powers):
        """
        在更新阶段使用：计算给定动作的 log_prob 和 entropy
        """
        channel_logits, power_mean, power_std = self.forward(obs)

        # 1. 信道分布 (Categorical)
        channel_dist = torch.distributions.Categorical(logits=channel_logits)
        channel_log_probs = channel_dist.log_prob(action_channels)
        channel_entropy = channel_dist.entropy()

        # 2. 功率分布 (Normal)
        power_dist = torch.distributions.Normal(power_mean, power_std)
        power_log_probs = power_dist.log_prob(action_powers)
        power_entropy = power_dist.entropy()

        # 合并 log_prob
        total_log_prob = channel_log_probs.sum(dim=-1) + power_log_probs.sum(dim=-1)
        total_entropy = channel_entropy.sum(dim=-1) + power_entropy.sum(dim=-1)

        return total_log_prob, total_entropy


class MAPPOLagrangianFramework:
    """
    【核心】MAPPO + Lagrangian + Attention + Auxiliary Framework
    """

    def __init__(self, config):
        self.config = config

        # 【核心修正】实例化环境，以便进行交互和获取维度
        self.env = SatTerrestrialEnvironment(config)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 从 config 中提取参数
        params = config['model_params']['mappo_lagrangian']

        self.lr_actor = params['actor_lr']
        self.lr_critic = params['critic_lr']
        self.lr_dual = params['dual_lr']
        self.gamma = config['gamma']
        self.gae_lambda = params['gae_lambda']
        self.clip_ratio = params['clip_ratio']
        self.entropy_coef = params['entropy_coef']
        self.aux_coef = params['aux_coef']
        self.cost_limit = params['cost_limit']
        self.save_dir = params['save_dir']
        self.batch_size = config['batch_size']

        # 确保保存目录存在
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # 环境相关维度 (使用 config 中的最大值以支持 Padding)
        self.max_users = config['max_users_per_gnb']
        self.max_gnbs = config['max_num_gnbs']
        self.num_channels = config['num_channels']

        # 确定观测维度 (需与 myenv.py 一致)
        # obs = [local_ch, adjusted_ch, sat_int, sat_ch, prev_act, pred]
        # max_users * 4 + num_channels + max_gnbs
        self.local_obs_dim = (
                self.max_users * 4 + self.num_channels + self.max_gnbs
        )

        # --- 1. 初始化网络 ---
        # 创建 num_gnbs 个 Actor (独立参数)
        self.agents = [
            MAPPOActor(self.local_obs_dim, self.num_channels, self.max_users, hidden_dim=params['actor_hidden_dim']).to(
                self.device)
            for _ in range(config['num_gnbs'])  # 这里用 config['num_gnbs'] 而不是 max_gnbs，或者根据实际需求调整
        ]

        # Global Critic (Attention Based)
        self.global_critic = AttentionCritic(
            obs_dim=self.local_obs_dim,
            hidden_dim=params['critic_hidden_dim'],
            aux_output_dim=self.max_gnbs  # 预测每个 gNB 的负载
        ).to(self.device)

        # --- 2. 优化器 ---
        self.actor_optimizers = [optim.Adam(a.parameters(), lr=self.lr_actor) for a in self.agents]
        self.critic_optimizer = optim.Adam(self.global_critic.parameters(), lr=self.lr_critic)

        # --- 3. 创新点：拉格朗日乘子 ---
        self.log_lambda = torch.zeros(1, requires_grad=True, device=self.device)
        self.dual_optimizer = optim.Adam([self.log_lambda], lr=self.lr_dual)

        # --- 4. 存储器 ---
        self.memory = PPOMemory(batch_size=config['batch_size'], gamma=self.gamma, gae_lambda=self.gae_lambda,
                                device=self.device)

        # 临时存储单步数据
        self.temp_obs = []
        self.temp_actions_ch = []
        self.temp_actions_pow = []
        self.temp_probs = []

        # 不使用 Reward Scaling，统一使用原始奖励

    def select_actions(self, global_state, use_exploration=True):
        """
        与环境交互的接口。
        在此处调用 self.env.get_local_observation 获取观测。
        """
        actions = {}
        powers = {}

        # 清空临时存储
        self.temp_obs = []
        self.temp_actions_ch = []
        self.temp_actions_pow = []
        self.temp_probs = []

        current_gnbs = self.env.num_gnbs

        with torch.no_grad():
            for gnb_idx in range(current_gnbs):
                # 【核心交互】调用环境获取观测
                obs = self.env.get_local_observation(gnb_idx, global_state, self.max_gnbs, self.max_users)
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

                # Actor 前向
                ch_logits, pow_mean, pow_std = self.agents[gnb_idx](obs_tensor)

                # 采样
                # 1. Channel
                ch_dist = torch.distributions.Categorical(logits=ch_logits)
                ch_action = ch_dist.sample()  # (1, num_users)
                ch_log_prob = ch_dist.log_prob(ch_action)

                # 2. Power
                if not use_exploration:
                    pow_action = pow_mean
                    pow_log_prob = torch.zeros_like(pow_mean)
                else:
                    pow_dist = torch.distributions.Normal(pow_mean, pow_std)
                    pow_action = pow_dist.sample()
                    pow_action = torch.clamp(pow_action, 0, 1)  # Clip [0, 1]
                    pow_log_prob = pow_dist.log_prob(pow_action)

                # 3. 物理功率映射 (0-1 -> min-max)
                real_power = pow_action.cpu().numpy()[0] * (
                            self.env.power_max - self.env.power_min) + self.env.power_min

                # 4. 输出掩码：只保留真实用户数的部分
                current_users = self.env.users_per_gnb
                actions[gnb_idx] = ch_action.cpu().numpy()[0][:current_users]
                powers[gnb_idx] = real_power[:current_users]

                # 5. 存储 Tensor 数据用于 Memory (存全尺寸的数据，方便Batch处理)
                self.temp_obs.append(obs)
                self.temp_actions_ch.append(ch_action.cpu().numpy()[0])
                self.temp_actions_pow.append(pow_action.cpu().numpy()[0])
                self.temp_probs.append((ch_log_prob.sum() + pow_log_prob.sum()).cpu().item())

        return actions, powers

    def store_transition(self, reward, cost, done, aux_target):
        """
        将这一步的转换存入 Memory。
        需要传入 reward, cost, done, 以及辅助任务的真实标签(如prediction)。
        """
        # 构建 Critic 需要的序列输入 (Num_Agents, Obs_Dim)
        obs_list = np.array(self.temp_obs)

        # 计算当前的 Value (用于 Log)
        obs_tensor = torch.FloatTensor(obs_list).unsqueeze(0).to(self.device)
        with torch.no_grad():
            value, _ = self.global_critic(obs_tensor)
            value = value.item()

        self.memory.store(
            state=obs_list,  # 在 MAPPO 中，这里存所有 Agent 的观测集合作为 State
            obs=obs_list,
            action=(np.array(self.temp_actions_ch), np.array(self.temp_actions_pow)),
            prob=np.array(self.temp_probs),
            val=value,
            reward=reward,
            cost=cost,
            done=done,
            aux_target=aux_target  # 真实标签，例如下一时刻的负载
        )

    def train_step(self):
        """
        MAPPO + Lagrangian 更新主循环 (修正版)
        """
        # 只有当 memory 存满 batch_size 时才更新
        if len(self.memory.rewards) < self.batch_size:
            return

        next_value = 0
        current_lambda = torch.exp(self.log_lambda).item()

        # --- 1. 准备数据 ---
        # 直接使用原始奖励，不再做 Reward Scaling
        states, obs, all_ch_actions, all_pow_actions, old_probs, values, raw_rewards, costs, dones, aux_targets, batches = self.memory.generate_batches()

        # --- 2. 计算 GAE (使用原始奖励，与存储时 Critic 的 value 保持一致) ---
        returns, advantages = self.memory.compute_gae(
            next_value=0,
            current_lambda=current_lambda,
            external_rewards=None  # 使用原始奖励
        )

        # --- 3. 数据转 Tensor ---
        states = torch.FloatTensor(states).to(self.device)
        aux_targets = torch.FloatTensor(aux_targets).to(self.device)
        old_probs = torch.FloatTensor(old_probs).to(self.device)

        # 使用原始奖励计算的 returns 和 advantages
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)

        # 归一化 Advantage (对收敛极重要)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_critic_loss_val = 0
        total_actor_loss_val = 0

        # --- 4. PPO 更新循环 ---
        ppo_epochs = 1
        for _ in range(ppo_epochs):
            for batch_idx in batches:
                # 提取 Batch
                b_states = states[batch_idx]
                b_aux = aux_targets[batch_idx]

                # 【关键】这里使用的是缩放后的 returns 和 advantages
                b_adv = advantages[batch_idx]
                b_ret = returns[batch_idx]

                b_old_probs = old_probs[batch_idx]

                # === Update Critic ===
                cur_values, cur_aux_preds = self.global_critic(b_states)
                cur_values = cur_values.squeeze()

                # 使用 SmoothL1Loss
                value_loss = F.smooth_l1_loss(cur_values, b_ret)
                aux_loss = F.mse_loss(cur_aux_preds, b_aux)

                total_critic_loss = value_loss + self.aux_coef * aux_loss

                self.critic_optimizer.zero_grad()
                total_critic_loss.backward()
                self.critic_optimizer.step()
                total_critic_loss_val += total_critic_loss.item()

                # === Update Actor ===
                b_ch_acts_np = all_ch_actions[batch_idx]
                b_pow_acts_np = all_pow_actions[batch_idx]

                batch_ch_acts = torch.LongTensor(b_ch_acts_np).to(self.device)
                batch_pow_acts = torch.FloatTensor(b_pow_acts_np).to(self.device)

                current_log_probs = []
                entropies = []

                for i in range(self.env.num_gnbs):
                    agent_obs = b_states[:, i, :]
                    agent_ch = batch_ch_acts[:, i, :]
                    agent_pow = batch_pow_acts[:, i, :]

                    l_prob, ent = self.agents[i].evaluate(agent_obs, agent_ch, agent_pow)
                    current_log_probs.append(l_prob)
                    entropies.append(ent)

                # Joint Policy Log Prob
                current_log_probs = torch.stack(current_log_probs, dim=1).sum(dim=1)
                entropy = torch.stack(entropies, dim=1).mean()

                # Ratio
                b_old_joint_probs = b_old_probs.sum(dim=1)
                ratio = torch.exp(current_log_probs - b_old_joint_probs)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * b_adv

                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

                for opt in self.actor_optimizers:
                    opt.zero_grad()
                actor_loss.backward()
                for opt in self.actor_optimizers:
                    opt.step()

                total_actor_loss_val += actor_loss.item()

        # --- 5. 更新拉格朗日乘子 ---
        # 注意：这里计算 Cost 均值时，使用的是 memory 中的原始 Cost (0~1之间)，这是对的
        # 不需要对 Cost 进行缩放，因为 Lambda 会自己适应
        current_avg_cost = np.mean(self.memory.costs)

        dual_loss = -self.log_lambda * (current_avg_cost - self.cost_limit)

        self.dual_optimizer.zero_grad()
        dual_loss.backward()
        self.dual_optimizer.step()

        # 【修正4】增加 Lambda 上限保护 (防止初期梯度爆炸)
        with torch.no_grad():
            self.log_lambda.clamp_(max=3.0)  # max=3.0 意味着 lambda max ≈ 20.0

        # 清空 Memory
        self.memory.clear()

        return {
            "loss_critic": total_critic_loss_val / (ppo_epochs * len(batches)),
            "loss_actor": total_actor_loss_val / (ppo_epochs * len(batches)),
            "lambda": current_lambda,
            "avg_cost": current_avg_cost
        }

    def save_models(self, filename):
        """
        保存模型
        filename: 完整的文件路径 (例如: checkpoints/mappo/best_model.pth)
        """
        # 【修正】直接使用传入的 filename，不要再拼接 self.save_dir
        # 如果 filename 只是文件名（不含路径），这行代码才需要 os.path.join
        # 但 trainer.py 传进来的是完整路径，所以直接用

        # 为了兼容性，可以判断一下 filename 是否包含路径分隔符
        if os.path.sep not in filename:
            full_path = os.path.join(self.save_dir, filename)
        else:
            full_path = filename

        # 确保父目录存在 (双重保险)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        checkpoint = {
            'global_critic': self.global_critic.state_dict(),
            'actors': [agent.state_dict() for agent in self.agents],
            'log_lambda': self.log_lambda
        }
        torch.save(checkpoint, full_path)
        print(f"Models saved to {full_path}")

    def load_models(self, filename):
        """
        加载模型
        filename: 完整的文件路径
        """
        if os.path.sep not in filename:
            full_path = os.path.join(self.save_dir, filename)
        else:
            full_path = filename

        if not os.path.exists(full_path):
            print(f"Warning: Model file {full_path} not found.")
            return

        checkpoint = torch.load(full_path, map_location=self.device)
        self.global_critic.load_state_dict(checkpoint['global_critic'])
        for i, agent in enumerate(self.agents):
            agent.load_state_dict(checkpoint['actors'][i])

        if 'log_lambda' in checkpoint:
            with torch.no_grad():
                self.log_lambda.copy_(checkpoint['log_lambda'])

        print(f"Models loaded from {full_path}")