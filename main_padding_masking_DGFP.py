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
import datetime

from env.myenv import SatTerrestrialEnvironment

from model.mymodel import SatTerrestrialHPGAMARLFramework
from model.dqn import DQN_Framework
from model.gdfp import GDFP_Policy
from model.randompolicy import RandomPolicy
from model.IBR import GameTheory_IBR_Policy
from model.qmix import QMIX_Framework


# 【新增】定义一个通用的日志记录函数
def log_experiment(description, mode, config, results):
    """将实验的描述、配置和结果记录到日志文件中"""
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write(f"实验时间: {timestamp}\n")
        f.write(f"更新描述: {description}\n")
        f.write(f"运行模式: {mode.upper()}\n")
        f.write("--- 配置参数 ---\n")
        # 记录关键配置，排除一些不重要的字段
        for key, value in config.items():
            if key not in ['mode']:
                f.write(f"  {key}: {value}\n")
        f.write("--- 性能指标 ---\n")
        for key, value in results.items():
            f.write(f"  {key}: {value}\n")
        f.write("=" * 60 + "\n\n")
    print(f"\n实验结果已成功记录到文件: {LOG_PATH}")


def set_seed(seed):
    """
    固定所有相关的随机数种子，以确保实验的可复现性。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 适用于多GPU情况
        # 确保CUDA的卷积操作是确定性的，可能会牺牲一些性能
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    SEED = 42  # 您可以选择任何整数作为种子
    set_seed(SEED)


    # ==========================================================================
    # 【新增】主控制逻辑：训练或测试
    # ==========================================================================
    config = {
        'mode': 'test',  # 'train' 或 'test'
        'model_name':'gdfp',  # marl,gdfp,randompolicy,dqn,IBR,qmix
        'num_gnbs': 7, # 建议最大值7或者19    上一次选3  实际基站数
        'users_per_gnb': 16,  # 建议最大值为16  上一次选10 实际用户密度
        'num_channels': 10,  # 不改，定义为10。20MHz是一个非常标准的信道带宽
        'terrestrial_qos_mbps': 10,
        'satellite_qos_mbps': 20,
        'buffer_size': 500000,
        'gamma': 0.99,
        'batch_size': 256,
        'prediction_noise_std': 0.1,

        'training_episode': 5000,
        'testing_episode':200,
        'steps_per_episode':25,

        # 训练全尺寸模型
        'max_gnbs': 7,  # 最大基站数
        'max_users_per_gnb': 16,  # 最大用户密度




    "model_params": {
        "marl":{
            'critic_hidden_dim': 256,
            'actor_hidden_dim': 128,
            'critic_lr': 1e-4,
            'actor_lr': 3e-4,

            'save_dir': 'checkpoints/marl/'
        },
        "dqn": {
            'critic_hidden_dim': 256,
            'actor_hidden_dim': 128,
            'critic_lr': 1e-4,
            'actor_lr': 3e-4,

            'dqn_hidden_dim': 128,  # DQN 网络的隐藏层维度
            'dqn_lr': 3e-4,  # DQN 网络的学习率
            'num_power_levels': 5,  # 功率离散化等级
            'epsilon_start': 0.95,  # 探索率初始值
            'epsilon_decay': 0.9995,  # 探索率衰减因子
            'epsilon_min': 0.05,  # 探索率最小值
            'target_update_freq': 100,  # Target网络同步的频率（按step计算）
            # --- 5. 文件路径 ---
            'save_dir':'checkpoints/dqn/'
            # 'dqn_model_path': "checkpoints/dqn/best_dqn_model.pth",
            # 'reward_history_path': "checkpoints/dqn/reward_history_dqn.csv",
        },
        "gdfp": {
            'save_dir': 'checkpoints/gdfp/',
            'critic_hidden_dim': 256,
            'actor_hidden_dim': 128,
            'critic_lr': 1e-4,
            'actor_lr': 3e-4,
        },
        "randompolicy":{
            'save_dir': 'checkpoints/randompolicy/',
            'critic_hidden_dim': 256,
            'actor_hidden_dim': 128,
            'critic_lr': 1e-4,
            'actor_lr': 3e-4,

        },
        "IBR": {
            'save_dir': 'checkpoints/IBR/',
            'iterations': 10,  # IBR 算法的内部迭代次数
            'critic_hidden_dim': 256,
            'actor_hidden_dim': 128,
            'critic_lr': 1e-4,
            'actor_lr': 3e-4,
        },
        'qmix': {
            'critic_hidden_dim': 256,
            'actor_hidden_dim': 128,
            'critic_lr': 1e-4,
            'actor_lr': 3e-4,

            'lr': 5e-4,
            'hidden_dim': 64,
            'num_power_levels': 5,
            'epsilon_start': 1.0,
            'epsilon_decay': 0.9998,
            'epsilon_min': 0.05,
            'target_update_tau': 0.005,
            'save_dir': './checkpoints/QMIX/'
        },

    }

    }


    # 【新增】在程序开始时，获取本次运行的更新描述
    update_description = input("请输入本次代码/配置更新的内容: ")

    # if config['model_name'] == 'marl':
    #     framework = SatTerrestrialHPGAMARLFramework(config)
    # elif config['model_name'] == 'dqn':
    #     framework =  DQN_Framework(config)  # 新增
    # else:
    #     # 对于非训练的策略，我们只需要环境
    #     env = SatTerrestrialEnvironment(config)
    #     if config['model_name'] == 'random':
    #         policy = RandomPolicy(env)
        # elif config['model_name'] == 'gd_fp':
        #     policy = GDFP_Policy(env)


    # 【新增】为测试结果定义一个动态的文件名
    TEST_REWARD_PATH = f"result/test_rewards_{config['model_name']}.csv"

    framework = SatTerrestrialHPGAMARLFramework(config)

    print(f"星地融合网络配置:")
    print(f"- 运行模式: {config['mode']}")
    print(f"- 设备: {framework.device}")
    print(f"- gNB数量: {config['num_gnbs']}")
    print(f"- 每个gNB用户数: {config['users_per_gnb']}")
    print(f"- 局部观测维度 (含预测): {framework.local_obs_dim}")
    print(f"- 全局观测维度 (含预测): {framework.global_obs_dim}")
    print(f"- PRB利用率预测噪声标准差: {config['prediction_noise_std']}")



    if config['model_name'] == 'gdfp':
        model_params = config['model_params']['gdfp']
        save_dir = model_params['save_dir']
        MODEL_PATH = save_dir + f"best_model_{config['model_name']}.pth"
        REWARD_PATH = save_dir + f"reward_history_{config['model_name']}_{config['mode']}.csv"
        LOG_PATH = save_dir + f"experiment_log_{config['model_name']}_{config['mode']}.txt"

        # 对于所有非MARL，进入统一的测试循环
        print(f"\n--- 开始对策略 '{config['model_name']}' 进行测试 ---")

        test_episodes = config['testing_episode']
        total_test_reward = 0
        total_test_se = 0
        total_qos_violations = 0

        env = SatTerrestrialEnvironment(config)
        policy = GDFP_Policy(env)

        # --- 1. 定义要测试的用户密度列表 ---
        user_densities_to_test = [4, 8, 12, 16]

        for density in user_densities_to_test:
            print(f"\n--- 正在测试 users_per_gnb = {density} ---")

            # a) 为当前密度创建一个新的测试配置和环境
            test_config = config.copy()
            test_config['users_per_gnb'] = density
            test_env = SatTerrestrialEnvironment(test_config)

            # b) 【关键】为当前环境创建一个新的策略对象
            policy = GDFP_Policy(test_env)

            # c) 初始化当前密度的测试统计变量
            test_episodes = config['testing_episode']
            test_reward_history = []
            total_test_reward, total_test_se, total_qos_violations = 0, 0, 0

            # d) 运行内部测试循环
            for episode in range(test_episodes):
                state = test_env.reset()
                episode_reward = 0

                for step in range(config['steps_per_episode']):
                    # 调用策略进行决策
                    actions, powers = policy.select_actions(state)

                    # 与环境交互
                    next_state, reward, _, qos_violations, spectrum_efficiency = test_env.step(actions, powers)

                    # 累加各项性能指标
                    total_test_reward += reward
                    total_test_se += spectrum_efficiency
                    total_qos_violations += qos_violations
                    episode_reward += reward
                    state = next_state

                # 在每个回合结束后，计算该回合的平均奖励并记录
                avg_episode_reward = episode_reward / config['steps_per_episode']
                test_reward_history.append(avg_episode_reward)


            # 计算并打印平均性能
            avg_reward = total_test_reward / (test_episodes * config['steps_per_episode'])
            avg_se = total_test_se / (test_episodes * config['steps_per_episode'])
            avg_qos_violations = total_qos_violations / (test_episodes * config['steps_per_episode'])

            print("\n--- 测试结果 ---")
            print(f"平均奖励: {avg_reward:.4f}")
            print(f"平均频谱效率 (bps/Hz): {avg_se:.4f}")
            print(f"平均每步QoS违规用户数: {avg_qos_violations:.4f}")

    if config['model_name'] == 'gdfp':
        print(f"\n--- 开始 GDFP 算法的泛化能力测试 ---")

        model_params = config['model_params']['gdfp']
        save_dir = model_params['save_dir']
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # --- 1. 定义要测试的泛化维度列表 ---
        gnb_counts_to_test = [3, 5, 7]
        user_densities_to_test = [16]

        # --- 2. 外层循环：遍历不同的基站数量 ---
        for gnb_count in gnb_counts_to_test:
            # --- 3. 内层循环：遍历不同的用户密度 ---
            for user_density in user_densities_to_test:
                print(f"\n--- 正在测试: num_gnbs = {gnb_count}, users_per_gnb = {user_density} ---")

                # a) 为当前配置创建新的测试环境
                test_config = config.copy()
                test_config['num_gnbs'] = gnb_count
                test_config['users_per_gnb'] = user_density
                test_env = SatTerrestrialEnvironment(test_config)

                # b) 【关键】为当前环境创建一个新的策略对象
                policy = GDFP_Policy(test_env)

                # c) 初始化当前配置的测试统计变量
                test_episodes = config['testing_episode']
                test_reward_history = []
                total_test_reward, total_test_se, total_qos_violations = 0, 0, 0

                # d) 运行内部测试循环
                for episode in range(test_episodes):
                    state = test_env.reset()
                    episode_reward = 0

                    for step in range(config['steps_per_episode']):
                        actions, powers = policy.select_actions(state)
                        next_state, reward, _, qos_violations, spectrum_efficiency = test_env.step(actions, powers)

                        # 累加各项性能指标
                        total_test_reward += reward
                        total_test_se += spectrum_efficiency
                        total_qos_violations += qos_violations
                        episode_reward += reward
                        state = next_state

                    avg_episode_reward = episode_reward / config['steps_per_episode']
                    test_reward_history.append(avg_episode_reward)

                # e) 保存并打印当前配置的结果
                REWARD_PATH = save_dir + f"reward_history_{config['model_name']}_{config['mode']}_gnbs_{gnb_count}_users_{user_density}.csv"
                np.savetxt(REWARD_PATH, np.array(test_reward_history), delimiter=",")

                total_steps = test_episodes * config['steps_per_episode']
                avg_reward = total_test_reward / total_steps
                avg_se = total_test_se / total_steps
                avg_qos_violations = total_qos_violations / total_steps

                print(f"--- GDFP (gnbs={gnb_count}, users={user_density}) 测试结果 ---")
                print(f"  平均奖励: {avg_reward:.4f}")
                print(f"  平均频谱效率 (bps/Hz): {avg_se:.4f}")
                print(f"  平均每步QoS违规用户数: {avg_qos_violations:.4f}")

    if config['model_name'] == 'randompolicy':
        model_params = config['model_params']['randompolicy']
        save_dir = model_params['save_dir']
        MODEL_PATH = save_dir + f"best_model_{config['model_name']}.pth"
        REWARD_PATH = save_dir + f"reward_history_{config['model_name']}_{config['mode']}.csv"
        LOG_PATH = save_dir + f"experiment_log_{config['model_name']}_{config['mode']}.txt"

        # 对于所有非MARL，进入统一的测试循环
        print(f"\n--- 开始对策略 '{config['model_name']}' 进行测试 ---")

        test_episodes = config['testing_episode']
        total_test_reward = 0
        total_test_se = 0
        total_qos_violations = 0

        env = SatTerrestrialEnvironment(config)
        # policy = GDFP_Policy(env)
        policy = RandomPolicy(env)

        for episode in range(test_episodes):
            state = env.reset()  # 增加
            episode_reward = 0

            for step in range(config['steps_per_episode']):
                # 【关键】在测试时关闭探索
                # 统一的决策接口
                # if config['model_name'] == 'marl':
                #     actions, powers = framework.hierarchical_decision(state, use_exploration=False)
                #     next_state, reward, _, qos_violations, spectrum_efficiency = framework.env.step(actions, powers)
                # else:
                #     # 修正env.reset()使其能返回state
                #     if 'env' not in locals() and 'env' not in globals():
                #         env = SatTerrestrialEnvironment(config)
                #
                if episode == 0 and step == 0:
                    state = env._build_state(env.get_channel_gains(), {}, {}, {}, None, 0, 0)

                actions, powers = policy.select_actions(state)
                next_state, reward, _, qos_violations, spectrum_efficiency = env.step(actions, powers)

                total_test_reward += reward
                total_test_se += spectrum_efficiency
                total_qos_violations += qos_violations

                state = next_state

            if (episode + 1) % 100 == 0:
                print(f"已完成 {episode + 1}/{test_episodes} 个测试回合...")

        # 计算并打印平均性能
        avg_reward = total_test_reward / (test_episodes * config['steps_per_episode'])
        avg_se = total_test_se / (test_episodes * config['steps_per_episode'])
        avg_qos_violations = total_qos_violations / (test_episodes * config['steps_per_episode'])

        print("\n--- 测试结果 ---")
        print(f"平均奖励: {avg_reward:.4f}")
        print(f"平均频谱效率 (bps/Hz): {avg_se:.4f}")
        print(f"平均每步QoS违规用户数: {avg_qos_violations:.4f}")

    if config['model_name'] == 'IBR':
        model_params = config['model_params']['IBR']
        save_dir = model_params['save_dir']
        MODEL_PATH = save_dir + f"best_model_{config['model_name']}.pth"
        REWARD_PATH = save_dir + f"reward_history_{config['model_name']}_{config['mode']}.csv"
        LOG_PATH = save_dir + f"experiment_log_{config['model_name']}_{config['mode']}.txt"

        # 对于所有非MARL，进入统一的测试循环
        print(f"\n--- 开始对策略 '{config['model_name']}' 进行测试 ---")

        test_episodes = config['testing_episode']
        total_test_reward = 0
        total_test_se = 0
        total_qos_violations = 0

        env = SatTerrestrialEnvironment(config)
        policy = GameTheory_IBR_Policy(env, iterations=config['model_params']['IBR']['iterations'])

        # 3. 运行测试循环
        for episode in range(test_episodes):
            state = env.reset()
            episode_reward = 0

            for step in range(config['steps_per_episode']):
                # 【关键】调用 IBR 策略的决策函数
                # IBR 算法在内部进行迭代优化，然后返回一个确定的动作
                actions, powers = policy.select_actions(state, use_exploration=False)

                # 与环境交互
                next_state, reward, _, qos_violations, spectrum_efficiency = env.step(actions, powers)

                # 累加各项性能指标
                total_test_reward += reward
                total_test_se += spectrum_efficiency
                total_qos_violations += qos_violations
                episode_reward += reward
                state = next_state

            # 在每个回合结束后，计算该回合的平均奖励并记录
            avg_episode_reward = episode_reward / config['steps_per_episode']
            # test_reward_history.append(avg_episode_reward)

            if (episode + 1) % 100 == 0:
                print(f"  已完成 {episode + 1}/{test_episodes} 个测试回合...")

        # 计算并打印平均性能
        avg_reward = total_test_reward / (test_episodes * config['steps_per_episode'])
        avg_se = total_test_se / (test_episodes * config['steps_per_episode'])
        avg_qos_violations = total_qos_violations / (test_episodes * config['steps_per_episode'])

        print("\n--- 测试结果 ---")
        print(f"平均奖励: {avg_reward:.4f}")
        print(f"平均频谱效率 (bps/Hz): {avg_se:.4f}")
        print(f"平均每步QoS违规用户数: {avg_qos_violations:.4f}")


    # ==========================================================================
    # 【QMIX 训练分支】
    # ==========================================================================
    elif config['mode'] == 'train' and config['model_name'] == 'qmix':
        model_params = config['model_params']['qmix']
        save_dir = model_params['save_dir']
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        MODEL_PATH = save_dir + f"best_model_{config['model_name']}.pth"
        REWARD_PATH = save_dir + f"reward_history_{config['model_name']}_{config['mode']}.csv"
        LOG_PATH = save_dir + f"experiment_log_{config['model_name']}_{config['mode']}.txt"

        print("\n--- 开始 QMIX 算法训练 ---")
        qmix_framework = QMIX_Framework(config)

        best_avg_reward = -float('inf')
        reward_history = []

        for episode in range(config['training_episode']):
            state = qmix_framework.env.reset()
            episode_reward = 0

            # 初始化所有agent的GRU隐藏状态
            hidden_states = [torch.zeros(1, qmix_framework.hidden_dim).to(qmix_framework.device)
                             for _ in range(qmix_framework.num_agents)]
            for step in range(config['steps_per_episode']):
                actions, powers, joint_actions, next_hidden_states = qmix_framework.select_actions(state, hidden_states)
                next_state, reward, _, _, _ = qmix_framework.env.step(actions, powers)
                # 将经验存入Replay Buffer
                qmix_framework.replay_buffer.push(state, joint_actions, reward, next_state, False)
                # 训练和更新
                if len(qmix_framework.replay_buffer) > config['batch_size']:
                    qmix_framework.train_step()
                    qmix_framework.update_target_networks(tau=model_params['target_update_tau'])

                state = next_state
                hidden_states = next_hidden_states  # 更新隐藏状态
                episode_reward += reward

            # 记录和保存
            avg_reward = episode_reward / config['steps_per_episode']
            reward_history.append(avg_reward)

            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                qmix_framework.save_models(MODEL_PATH)
                print(f"*** Episode {episode}: New best reward {best_avg_reward:.2f}, QMIX model saved. ***")

            if (episode + 1) % 50 == 0:
                avg_50 = np.mean(reward_history[-50:]) if reward_history else 0.0
                print(f"Episode {episode}: Avg Reward(50) = {avg_50:.2f}, Epsilon = {qmix_framework.epsilon:.3f}")

        np.savetxt(REWARD_PATH, np.array(reward_history), delimiter=",")
        print(f"\nQMIX训练完成! Reward历史已保存到 {REWARD_PATH}")

    # ==========================================================================
    # 【QMIX 测试分支】
    # ==========================================================================

    # elif config['mode'] == 'test' and config['model_name'] == 'qmix':
    #     model_params = config['model_params']['qmix']
    #     save_dir = model_params['save_dir']
    #     MODEL_PATH = save_dir + f"best_model_{config['model_name']}.pth"
    #     REWARD_PATH = save_dir + f"reward_history_{config['model_name']}_{config['mode']}.csv"
    #
    #     print("\n--- 开始 QMIX 算法的用户密度泛化能力测试 ---")
    #     qmix_framework = QMIX_Framework(config)
    #     qmix_framework.load_models(MODEL_PATH)
    #
    #     # 设置为评估模式
    #     for agent_net in qmix_framework.agent_networks:
    #         agent_net.eval()
    #     qmix_framework.mixing_network.eval()
    #
    #     test_reward_history = []
    #     total_test_reward, total_test_se, total_qos_violations = 0, 0, 0
    #
    #     # --- 2. 【新增】定义要测试的用户密度列表 ---
    #     user_densities_to_test = [4, 8, 12, 16]
    #     for density in user_densities_to_test:
    #         # a) 为当前密度创建一个新的测试配置和环境
    #         test_config = config.copy()
    #         test_config['users_per_gnb'] = density
    #         test_env = SatTerrestrialEnvironment(test_config)
    #
    #         # b) 【关键】将主框架的环境临时指向当前测试环境
    #         qmix_framework.env = test_env
    #
    #         # c) 初始化当前密度的测试统计变量
    #         test_episodes = config['testing_episode']
    #         test_reward_history = []
    #         total_test_reward, total_test_se, total_qos_violations = 0, 0, 0
    #
    #         # 5. 运行测试循环
    #         for episode in range(config['testing_episode']):
    #             state = qmix_framework.env.reset()
    #             episode_reward = 0
    #
    #             # 【关键】在每个 episode 开始时，必须重置 GRU 的隐藏状态
    #             hidden_states = [torch.zeros(1, qmix_framework.hidden_dim).to(qmix_framework.device)
    #                              for _ in range(qmix_framework.num_agents)]
    #
    #             for step in range(config['steps_per_episode']):
    #                 # 【关键】调用决策函数时，传入当前的 hidden_states 并关闭探索
    #                 actions, powers, _, next_hidden_states = qmix_framework.select_actions(
    #                     state, hidden_states, use_exploration=False
    #                 )
    #
    #                 # 与环境交互
    #                 next_state, reward, _, qos_violations, spectrum_efficiency = qmix_framework.env.step(actions, powers)
    #
    #                 # 累加各项性能指标
    #                 total_test_reward += reward
    #                 total_test_se += spectrum_efficiency
    #                 total_qos_violations += qos_violations
    #                 episode_reward += reward
    #
    #                 # 更新 state 和 hidden_states 以进行下一步循环
    #                 state = next_state
    #                 hidden_states = next_hidden_states
    #
    #             # 在每个回合结束后，计算该回合的平均奖励并记录
    #             avg_episode_reward = episode_reward / config['steps_per_episode']
    #             test_reward_history.append(avg_episode_reward)
    #
    #         # 6. 将每个回合的奖励历史保存到CSV文件
    #         np.savetxt(REWARD_PATH, np.array(test_reward_history), delimiter=",")
    #         print(f"\n测试完成! 每个Episode的平均Reward已保存到 {REWARD_PATH}")
    #
    #         # 7. 计算并打印总体的平均性能
    #         total_steps = config['testing_episode'] * config['steps_per_episode']
    #         avg_reward = total_test_reward / total_steps
    #         avg_se = total_test_se / total_steps
    #         avg_qos_violations = total_qos_violations / total_steps
    #
    #         print(f"\n--- QMIX 整体测试结果 ---")
    #         print(f"  平均奖励: {avg_reward:.4f}")
    #         print(f"  平均频谱效率 (bps/Hz): {avg_se:.4f}")
    #         print(f"  平均每步QoS违规用户数: {avg_qos_violations:.4f}")


    elif config['mode'] == 'test' and config['model_name'] == 'qmix':
        model_params = config['model_params']['qmix']
        save_dir = model_params['save_dir']
        MODEL_PATH = save_dir + f"best_model_{config['model_name']}.pth"
        REWARD_PATH = save_dir + f"reward_history_{config['model_name']}_{config['mode']}.csv"

        print("\n--- 开始 QMIX 算法的基站数量泛化能力测试 ---")

        qmix_framework = QMIX_Framework(config)
        qmix_framework.load_models(MODEL_PATH)


        # 设置为评估模式
        for agent_net in qmix_framework.agent_networks:
            agent_net.eval()
        qmix_framework.mixing_network.eval()

        test_reward_history = []
        total_test_reward, total_test_se, total_qos_violations = 0, 0, 0

        #########################
        # --- 2. 【新增】定义要测试的基站数量和用户密度列表 ---
        gnb_counts_to_test = [3, 5, 7]
        user_densities_to_test = config['max_users_per_gnb']   # 您也可以测试多种用户密度 [8, 12, 16]

        # --- 3. 【新增】外层循环，遍历不同的基站数量 ---
        for gnb_count  in gnb_counts_to_test:
            # a) 为当前密度创建一个新的测试配置和环境
            test_config = config.copy()
            test_config['users_per_gnb'] = user_densities_to_test
            test_config['num_gnbs'] = gnb_count
            test_env = SatTerrestrialEnvironment(test_config)

            # b) 【关键】将主框架的环境临时指向当前测试环境
            qmix_framework.env = test_env

            # c) 初始化当前密度的测试统计变量
            test_episodes = config['testing_episode']
            test_reward_history = []
            total_test_reward, total_test_se, total_qos_violations = 0, 0, 0

            # 5. 运行测试循环
            for episode in range(config['testing_episode']):
                state = qmix_framework.env.reset()
                episode_reward = 0

                # 【关键】在每个 episode 开始时，必须重置 GRU 的隐藏状态
                hidden_states = [torch.zeros(1, qmix_framework.hidden_dim).to(qmix_framework.device)
                                 for _ in range(qmix_framework.num_agents)]

                for step in range(config['steps_per_episode']):
                    # 【关键】调用决策函数时，传入当前的 hidden_states 并关闭探索
                    actions, powers, _, next_hidden_states = qmix_framework.select_actions(
                        state, hidden_states, use_exploration=False
                    )

                    # 与环境交互
                    next_state, reward, _, qos_violations, spectrum_efficiency = qmix_framework.env.step(actions, powers)

                    # 累加各项性能指标
                    total_test_reward += reward
                    total_test_se += spectrum_efficiency
                    total_qos_violations += qos_violations
                    episode_reward += reward

                    # 更新 state 和 hidden_states 以进行下一步循环
                    state = next_state
                    hidden_states = next_hidden_states

                # 在每个回合结束后，计算该回合的平均奖励并记录
                avg_episode_reward = episode_reward / config['steps_per_episode']
                test_reward_history.append(avg_episode_reward)

            # 6. 将每个回合的奖励历史保存到CSV文件
            np.savetxt(REWARD_PATH, np.array(test_reward_history), delimiter=",")
            print(f"\n测试完成! 每个Episode的平均Reward已保存到 {REWARD_PATH}")

            # 7. 计算并打印总体的平均性能
            total_steps = config['testing_episode'] * config['steps_per_episode']
            avg_reward = total_test_reward / total_steps
            avg_se = total_test_se / total_steps
            avg_qos_violations = total_qos_violations / total_steps

            print(f"\n--- QMIX 整体测试结果 ---")
            print(f"  平均奖励: {avg_reward:.4f}")
            print(f"  平均频谱效率 (bps/Hz): {avg_se:.4f}")
            print(f"  平均每步QoS违规用户数: {avg_qos_violations:.4f}")