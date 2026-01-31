# 星地融合频谱资源分配强化学习项目

## 一、项目概述

本项目研究卫星网络与地面蜂窝网络（如5G/6G）的频谱共享问题，通过强化学习算法优化以下决策：
- **信道分配**：为每个地面用户分配最佳通信信道
- **功率控制**：为每个地面用户分配最佳传输功率

## 二、目录结构

```
resourceallocation2/
├── main.py                    # 【主入口】程序入口，根据config选择模型
├── trainer.py                 # 训练器基类及实现
│
├── env/                       # 环境模块
│   ├── myenv.py              # 地面环境（使用myenv.py的观测定义）
│   └── myenv2.py             # 地面环境（简化版 reward）
│
├── model/                     # 算法模型模块
│   ├── mymodel.py            # 基础MARL框架（独立Actor + 共享Critic）
│   ├── mymodel260129.py      # 【当前使用】带辅助任务的MARL（mymodel改进版）
│   ├── mymodel260128.py      # mymodel260129的早期版本
│   ├── mymodel_graph_2601230.py  # 图注意力网络版本的MARL
│   ├── mappo_lagrangian.py   # MAPPO + Lagrangian约束优化
│   ├── dqn.py                # DQN（离散动作空间）
│   ├── gdfp.py               # GDFP启发式策略
│   ├── randompolicy.py       # 随机策略（基线）
│   ├── IBR.py                # 迭代注水算法（传统方法）
│   └── qmix.py               # QMIX算法
│
├── physic/                    # 物理层模块
│   └── physic.py             # 卫星轨道模型、信道模型
│
├── draw/                      # 可视化模块
│   ├── pic.py                # 绘图工具
│   ├── pic_tot.py            # 综合绘图
│   └── pic_opt.py            # 优化结果绘图
│
├── result/                    # 结果分析脚本
│   ├── SE vs testing episodes/
│   ├── SE vs transmit powers/
│   ├── QoS violation vs gNBs/
│   └── sensitive/            # 敏感性分析
│
├── backup/                    # 历史版本备份
│   ├── hpgamarl_v*.py        # 各版本的MARL实现
│   └── myenv.py              # 旧版环境定义
│
├── checkpoints/              # 模型保存目录（按模型名分）
│   ├── marl/
│   ├── dqn/
│   ├── mappo_lagrangian/
│   └── ...
│
└── log/                       # 实验日志
    └── experiment_log2.txt   # 实验记录
```

## 三、核心模块详解

### 3.1 环境模块 (`env/`)

| 文件 | 说明 | 区别 |
|------|------|------|
| `myenv.py` | 完整版环境，包含PRB利用率预测、Reward Shaping | 奖励包含：频谱效率 + QoS奖励 + 拥塞惩罚 |
| `myenv2.py` | 简化版环境 | 奖励仅使用原始频谱效率，惩罚交给Lagrangian处理 |

**环境观测向量结构** (`myenv.get_local_observation`)：
```
- padded_local_channels:          16维 (信道增益)
- congestion_adjusted_channels:   16维 (拥塞调整后的信道增益)
- padded_sat_interference:        16维 (卫星干扰)
- satellite_channels:             10维 (卫星信道占用)
- padded_prev_actions:            16维 (前序动作)
- padded_prediction:              7/19维 (PRB利用率预测)
总计: 81维 (max_users=16, num_channels=10, max_gnbs=7)
     或 93维 (max_gnbs=19)
```

### 3.2 训练器模块 (`trainer.py`)

| 类名 | 算法类型 | 说明 |
|------|----------|------|
| `BaseTrainer` | - | 基类，包含日志、模型保存/加载 |
| `OnPolicyTrainer` | PPO/MAPPO | 回合结束后更新 |
| `OffPolicyTrainer` | DQN/MARL | 每步更新，需ReplayBuffer |
| `HeuristicTrainer` | 启发式 | GDFP/RandomPolicy专用 |

### 3.3 算法模型模块 (`model/`)

#### 3.3.1 MARL系列（多智能体强化学习）

| 文件 | 算法 | 特点 |
|------|------|------|
| `mymodel.py` | 基础MADDPG风格 | 独立Actor + 共享Attention Critic |
| `mymodel260129.py` | **当前使用** | mymodel基础上增加辅助任务头（预测PRB利用率） |
| `mymodel260128.py` | 过渡版本 | mymodel260129的早期实现 |
| `mymodel_graph_2601230.py` | 图注意力网络 | 使用Graph Attention处理gNB间拓扑关系 |

**mymodel → mymodel260129 改进点**：
```
mymodel:
    Critic: AttentionCritic -> 输出 (Q值, 辅助预测)

mymodel260129:
    Critic: SatTerrestrialCritic(新设计)
           - Encoder: 独立特征提取
           - Attention: 多头注意力聚合
           - Q_head: 2*hidden_dim -> 1
           - Aux_head: hidden_dim -> max_gnbs (预测负载)
```

#### 3.3.2 MAPPO系列

| 文件 | 算法 | 特点 |
|------|------|------|
| `mappo_lagrangian.py` | MAPPO + Lagrangian | 使用Lagrangian乘子处理QoS约束 |

**MAPPO架构**：
```
- Actor: MAPPOActor (每个gNB独立)
  - Channel Head: 输出信道选择logits
  - Power Head: 输出功率均值(σ固定)

- Critic: AttentionCritic (全局共享)
  - SelfAttention: 聚合多智能体信息
  - Value Head: 输出V(s)
  - Aux Head: 预测下时刻PRB利用率

- Lagrangian: λ参数控制QoS约束
  - Loss = L_PPO + λ * (Cost - Limit)
```

#### 3.3.3 其他算法

| 文件 | 算法 | 类型 |
|------|------|------|
| `dqn.py` | DQN | 离散动作（信道选择） |
| `gdfp.py` | GDFP | 启发式（贪婪信道选择） |
| `randompolicy.py` | Random | 随机基线 |
| `IBR.py` | 迭代注水 | 传统优化算法 |
| `qmix.py` | QMIX | 值分解方法 |

## 四、运行方式

### 4.1 基本运行

```bash
python main.py
```

### 4.2 选择不同模型

修改 `main.py` 中的配置：

```python
'model_name': 'marl',           # mymodel260129 (当前)
'model_name': 'mappo_lagrangian', # MAPPO
'model_name': 'dqn',            # DQN
'model_name': 'gdfp',           # 启发式
'model_name': 'randompolicy',   # 随机基线
```

### 4.3 关键配置参数

```python
# 环境参数
'num_gnbs': 7 或 19,          # gNB数量
'users_per_gnb': 16,          # 每个gNB的用户数
'num_channels': 10,           # 信道数量
'terrestrial_qos_mbps': 10,   # 地面用户QoS阈值
'satellite_qos_mbps': 20,     # 卫星QoS阈值

# 训练参数
'batch_size': 64,             # 批大小
'trainging_episode': 4000,    # 训练轮数
'steps_per_episode': 25,      # 每轮步数
'gamma': 0.99,                # 折扣因子

# 模型维度（需与num_gnbs匹配）
'max_num_gnbs': 7 或 19,      # 模型支持的最大gNB数
'max_users_per_gnb': 16,      # 模型支持的最大用户数
```

## 五、名字相似模块的改进关系

### 5.1 mymodel 系列

```
mymodel.py (v1)
    ↓
mymodel260128.py (过渡版本)
    ↓
mymodel260129.py (当前使用)
    │
    └── mymodel_graph_2601230.py (并行分支，图注意力版本)
```

**改进点**：
1. 维度修复：观测维度从71→81（修复congestion_adjusted_channels）
2. Critic架构重构：新增SatTerrestrialCritic类
3. 辅助任务：增加负载预测头

### 5.2 myenv 系列

```
myenv.py (完整版，复杂reward)
    ↓
myenv2.py (简化版，简化reward)
```

**选择建议**：
- 快速实验：用 `myenv2.py` + `mappo_lagrangian.py`
- 完整实验：用 `myenv.py` + `mymodel260129.py`

### 5.3 main 系列

```
main.py (当前主入口)
main2.py (备用入口)
main260127.py (历史版本)
main_padding_masking_*.py (带padding/masking的变体)
```

## 六、常见问题排查

### 6.1 维度不匹配

```
错误: 'SatTerrestrialCritic' object has no attribute 'layer_norm'
解决: mymodel260129.py添加了layer_norm，确保代码已更新

错误: shapes cannot be multiplied (1x81 and 97x128)
解决: max_num_gnbs需与num_gnbs保持一致
```

### 6.2 奖励不提升

1. 检查 `local_obs_dim` 是否正确（4*max_users + num_channels + max_gnbs）
2. 确认 `reward_scale` 设置合理
3. 尝试调整学习率 `actor_lr`, `critic_lr`

### 6.3 Cost不收敛

- 调整 `cost_limit`（当前0.50）
- 调整 `dual_lr`（Lagrangian乘子学习率）
- 确认 `mymodel260129.py` 中 `beta_aux` 参数

## 七、关键文件清单

| 文件路径 | 功能 | 修改频率 |
|----------|------|----------|
| `main.py` | 入口配置 | 中 |
| `model/mymodel260129.py` | 核心MARL算法 | 高 |
| `env/myenv.py` | 环境定义 | 中 |
| `trainer.py` | 训练循环 | 低 |
| `model/mappo_lagrangian.py` | MAPPO算法 | 中 |

---
*文档生成时间: 2026-01-31*
