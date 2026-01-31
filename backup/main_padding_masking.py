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
        'model_name':'dqn',  # marl,gdfp,randompolicy,dqn
        'num_gnbs': 7, # 建议最大值7或者19    上一次选3  实际基站数
        'users_per_gnb': 16,  # 建议最大值为16  上一次选10 实际用户密度
        'num_channels': 10,  # 不改，定义为10。20MHz是一个非常标准的信道带宽
        'terrestrial_qos_mbps': 10,
        'satellite_qos_mbps': 20,
        'buffer_size': 500000,
        'gamma': 0.99,
        'batch_size': 256,
        'prediction_noise_std': 0.1,

        'trainging_episode': 5000,
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
            'save_dir': 'checkpoints/gdfp/'
        },
        "randompolicy":{
            'save_dir': 'checkpoints/randompolicy/'
        }

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

    if config['model_name'] == 'marl' and config['mode'] == 'train':
        # 定义存储地址
        model_params = config['model_params']['marl']
        save_dir = model_params['save_dir']
        # "checkpoints/best_maddpg_model.pth"
        MODEL_PATH = save_dir + f"best_model_{config['model_name']}.pth"
        # DQN_MODEL_PATH = "checkpoints/best_dqn_model.pth"  # 为DQN策略指定模型路径
        # REWARD_PATH = f"result/reward_history_{config['model_name']}.csv"
        REWARD_PATH =  save_dir + f"reward_history_{config['model_name']}_{config['mode']}.csv"
        LOG_PATH = save_dir + f"experiment_log_{config['model_name']}_{config['mode']}.txt"

        framework = SatTerrestrialHPGAMARLFramework(config)

        print("\n--- 开始marl训练 ---")
        training_episodes = config['trainging_episode']
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

            for step in range(config['steps_per_episode']):
                step_start_time = time.time()  # 记录单步总时间的开始
                # 1. 计时：决策过程 取消决策过程的及时
                start_time = time.time()

                # actions, powers = framework.hierarchical_decision(state)
                actions, powers = framework.hierarchical_decision(state)


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
            for key, total_time in time_stats.items():
                avg_time = total_time / config['steps_per_episode']
                # 打印毫秒(ms)为单位的时间，更直观
                print(f"- {key:<15}: {avg_time * 1000:.4f} ms")
            print("------------------------------------")

            avg_reward = episode_reward / config['steps_per_episode']
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

        # 【新增】训练结束后，记录结果
        final_results = {
            "最佳平均奖励": f"{best_avg_reward:.4f}",
            "总训练回合数": training_episodes
        }
        log_experiment(update_description, 'train', config, final_results)

    elif config['mode'] == 'test' and config['model_name']=='marl':
        print("\n--- 开始测试 ---")
        model_params = config['model_params']['marl']
        save_dir = model_params['save_dir']
        MODEL_PATH = save_dir + f"best_model_{config['model_name']}.pth"
        REWARD_PATH =  save_dir + f"reward_history_{config['model_name']}_{config['mode']}.csv"
        LOG_PATH = save_dir + f"experiment_log_{config['model_name']}_{config['mode']}.txt"

        # # 【新增】定义要测试的用户密度列表
        # user_densities_to_test = [4, 8, 12, 16]
        #
        # # 2. 初始化一个基于最大用户数的全尺寸框架，并加载模型
        # # 这个框架将作为我们唯一的“决策者”
        # print(f"加载基于 max_users_per_gnb={config['max_users_per_gnb']} 训练的模型...")
        # framework = SatTerrestrialHPGAMARLFramework(config)
        # framework.load_models(MODEL_PATH)
        # for agent in framework.agents:
        #     agent.actor.eval() # 将所有智能体设为评估模式
        #
        # test_episodes = config['testing_episode']
        #
        # test_reward_history = []
        # total_test_reward = 0
        # total_test_se = 0
        # total_qos_violations = 0
        # # 2. 循环遍历不同的用户密度
        # # 3. 循环遍历不同的用户密度
        # for density in user_densities_to_test:
        #     print(f"\n--- 正在测试 users_per_gnb = {density} ---")
        #
        #     # 4. 为当前密度创建一个新的测试环境
        #     test_config = config.copy()
        #     test_config['users_per_gnb'] = density
        #     test_env = SatTerrestrialEnvironment(test_config)
        #
        #     # 5. 【核心修改】将测试环境临时赋给主框架
        #     framework.env = test_env
        #
        #     # 6. 运行测试
        #     test_episodes = 200
        #     total_test_reward = 0
        #     total_test_se = 0
        #     total_qos_violations = 0
        #
        #     for episode in range(test_episodes):
        #         state = framework.env.reset()  # 使用当前测试环境的reset
        #
        #         for step in range(config['steps_per_episode']):
        #             # 7. 【核心修改】始终使用主框架进行决策
        #             # decision函数内部会自动调用 self.env (现在是test_env) 的方法
        #             # 从而正确地生成填充后的观测数据
        #             actions, powers = framework.hierarchical_decision(state, use_exploration=False)
        #
        #             # 8. 在当前测试环境中执行动作
        #             next_state, reward, _, qos_violations, spectrum_efficiency = framework.env.step(actions, powers)
        #
        #             total_test_reward += reward
        #             total_test_se += spectrum_efficiency
        #             total_qos_violations += qos_violations
        #             state = next_state
        #
        #     # 9. 打印当前密度下的测试结果
        #     total_steps = test_episodes * config['steps_per_episode']
        #     avg_reward = total_test_reward / total_steps
        #     avg_se = total_test_se / total_steps
        #     avg_qos_violations = total_qos_violations / total_steps
        #
        #     print(f"--- users_per_gnb = {density} 测试结果 ---")
        #     print(f"  平均奖励: {avg_reward:.4f}")
        #     print(f"  平均频谱效率 (bps/Hz): {avg_se:.4f}")
        #     print(f"  平均每步QoS违规用户数: {avg_qos_violations:.4f}")

     ########################################################################
        print("\n--- 开始多基站数泛化测试 ---")

        # 【新增】定义要测试的基站数量列表
        gnb_counts_to_test = [3, 5, 7]
        user_density_to_test = 16  # 固定用户密度进行测试

        # 1. 初始化并加载一个基于最大配置的全尺寸框架
        print(f"加载基于 max_gnbs={config['max_gnbs']} 训练的模型...")
        framework = SatTerrestrialHPGAMARLFramework(config)
        framework.load_models(MODEL_PATH)
        for agent in framework.agents:
            agent.actor.eval()

        # 2. 循环遍历不同的基站数量
        for count in gnb_counts_to_test:
            print(f"\n--- 正在测试 num_gnbs = {count} ---")

            # a) 创建对应的测试配置和环境
            test_config = config.copy()
            test_config['num_gnbs'] = count
            test_config['users_per_gnb'] = user_density_to_test  # 使用固定的用户密度
            test_env = SatTerrestrialEnvironment(test_config)

            # b) 将测试环境临时赋给主框架
            framework.env = test_env

            # 6. 运行测试
            test_episodes = 200
            total_test_reward = 0
            total_test_se = 0
            total_qos_violations = 0
            # 运行内部测试循环 (与之前的逻辑相同)
            for episode in range(test_episodes):
                state = framework.env.reset()  # 使用当前测试环境的reset

                for step in range(config['steps_per_episode']):
                    # 7. 【核心修改】始终使用主框架进行决策
                    # decision函数内部会自动调用 self.env (现在是test_env) 的方法
                    # 从而正确地生成填充后的观测数据
                    actions, powers = framework.hierarchical_decision(state, use_exploration=False)

                    # 8. 在当前测试环境中执行动作
                    next_state, reward, _, qos_violations, spectrum_efficiency = framework.env.step(actions, powers)

                    total_test_reward += reward
                    total_test_se += spectrum_efficiency
                    total_qos_violations += qos_violations
                    state = next_state
                # 9. 打印当前密度下的测试结果
                total_steps = test_episodes * config['steps_per_episode']
                avg_reward = total_test_reward / total_steps
                avg_se = total_test_se / total_steps
                avg_qos_violations = total_qos_violations / total_steps

            print(f"--- num_gnbs = {count} 测试结果 ---")
            print(f"  平均奖励: {avg_reward:.4f}")
            print(f"  平均频谱效率 (bps/Hz): {avg_se:.4f}")
            print(f"  平均每步QoS违规用户数: {avg_qos_violations:.4f}")

    if config['mode'] == 'train' and config['model_name'] == 'dqn':
        model_params = config['model_params']['dqn']
        save_dir = model_params['save_dir']
        MODEL_PATH = save_dir + f"best_model_{config['model_name']}.pth"
        REWARD_PATH = save_dir + f"reward_history_{config['model_name']}_{config['mode']}.csv"
        LOG_PATH = save_dir + f"experiment_log_{config['model_name']}_{config['mode']}.txt"

        framework = DQN_Framework(config)  # 新增

        print("\n--- 开始 DQN 联合分配算法训练 ---")
        dqn_framework = DQN_Framework(config)
        training_episodes = config['trainging_episode']
        best_avg_reward = -float('inf')
        reward_history = []
        global_step = 0
        for episode in range(training_episodes):
            state = dqn_framework.env.reset()
            episode_reward = 0
            for step in range(config['steps_per_episode']):
                global_step += 1
                actions, powers, joint_actions = dqn_framework.select_actions(state)
                next_state, reward, _, _, _ = dqn_framework.env.step(actions, powers)
                episode_reward += reward

                # 存入Replay Buffer的是联合动作索引

                dqn_framework.replay_buffer.push(state, joint_actions, reward, next_state, False)
                if len(dqn_framework.replay_buffer) > config['batch_size']:
                    dqn_framework.train_step()
                # 定期更新Target Network
                if global_step % model_params['target_update_freq'] == 0:
                    for agent in dqn_framework.agents:
                        agent.update_target_network()

                state = next_state

            avg_reward = episode_reward / config['steps_per_episode']

            reward_history.append(avg_reward)

            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                dqn_framework.save_models(MODEL_PATH)
                print(f"*** Episode {episode}: New best reward {best_avg_reward:.2f}, DQN model saved. ***")

            if episode % 100 == 0:
                avg_100 = np.mean(reward_history[-100:]) if reward_history else 0.0

                print(f"Episode {episode}: Avg Reward(100) = {avg_100:.2f}, Best Avg Reward = {best_avg_reward:.2f}")

        np.savetxt(REWARD_PATH, np.array(reward_history), delimiter=",")
        print(f"\nDQN训练完成! Reward历史已保存到 {REWARD_PATH}")



    elif config['mode'] == 'test' and config['model_name'] == 'dqn':
        # --- 1. 初始化一个基于最大用户数的全尺寸DQN框架 ---
        # 这个框架将作为我们唯一的“决策者”，并加载预训练模型
        print(f"\n--- 准备DQN泛化测试 ---")
        print(f"加载基于 max_users_per_gnb={config['max_users_per_gnb']} 训练的模型...")
        dqn_framework = DQN_Framework(config)

        model_params = config['model_params']['dqn']
        save_dir = model_params['save_dir']
        MODEL_PATH = save_dir + f"best_model_{config['model_name']}.pth"
        dqn_framework.load_models(MODEL_PATH)

        # 将所有智能体网络设置为评估模式
        for agent in dqn_framework.agents:
            agent.q_network.eval()

        # # --- 2. 【新增】定义要测试的用户密度列表 ---
        # user_densities_to_test = [4, 8, 12, 16]
        #
        # # --- 3. 【新增】循环遍历不同的用户密度 ---
        # for density in user_densities_to_test:
        #     print(f"\n--- 正在测试 users_per_gnb = {density} ---")
        #
        #     # a) 为当前密度创建一个新的测试配置和环境
        #     test_config = config.copy()
        #     test_config['users_per_gnb'] = density
        #     test_env = SatTerrestrialEnvironment(test_config)
        #
        #     # b) 【关键】将主框架的环境临时指向我们为当前密度创建的测试环境
        #     dqn_framework.env = test_env
        #
        #     # c) 初始化当前密度的测试统计变量
        #     test_episodes = config['testing_episode']
        #     test_reward_history = []
        #     total_test_reward = 0
        #     total_test_se = 0
        #     total_qos_violations = 0
        #
        #     # d) 运行内部测试循环
        #     for episode in range(test_episodes):
        #         state = dqn_framework.env.reset()  # 使用当前测试环境的reset
        #         episode_reward = 0
        #
        #         for step in range(config['steps_per_episode']):
        #             # 调用主框架进行决策，它内部会自动处理输入填充和输出掩码
        #             actions, powers, _ = dqn_framework.select_actions(state, use_exploration=False)
        #
        #             # 在当前测试环境中执行动作
        #             next_state, reward, _, qos_violations, spectrum_efficiency = dqn_framework.env.step(actions, powers)
        #
        #             # 累加各项性能指标
        #             total_test_reward += reward
        #             total_test_se += spectrum_efficiency
        #             total_qos_violations += qos_violations
        #             episode_reward += reward
        #             state = next_state
        #
        #         # 在每个回合结束后，计算该回合的平均奖励并记录
        #         avg_episode_reward = episode_reward / config['steps_per_episode']
        #         test_reward_history.append(avg_episode_reward)
        #
        #         if (episode + 1) % 100 == 0:
        #             print(f"  已完成 {episode + 1}/{test_episodes} 个测试回合...")
        #
        #     # e) 将当前密度的奖励历史保存到特定命名的CSV文件
        #     REWARD_PATH = save_dir + f"reward_history_{config['model_name']}_{config['mode']}_users_{density}.csv"
        #     np.savetxt(REWARD_PATH, np.array(test_reward_history), delimiter=",")
        #     print(f"  测试完成! 每个Episode的平均Reward已保存到 {REWARD_PATH}")
        #
        #     # f) 计算并打印当前密度下的总体平均性能
        #     total_steps = test_episodes * config['steps_per_episode']
        #     avg_reward = total_test_reward / total_steps
        #     avg_se = total_test_se / total_steps
        #     avg_qos_violations = total_qos_violations / total_steps
        #
        #     print(f"\n--- DQN users_per_gnb = {density} 整体测试结果 ---")
        #     print(f"  平均奖励: {avg_reward:.4f}")
        #     print(f"  平均频谱效率 (bps/Hz): {avg_se:.4f}")
        #     print(f"  平均每步QoS违规用户数: {avg_qos_violations:.4f}")

        #########################
        # --- 2. 【新增】定义要测试的基站数量和用户密度列表 ---
        gnb_counts_to_test = [3, 5, 7]
        user_densities_to_test = [16]  # 您也可以测试多种用户密度 [8, 12, 16]

        # --- 3. 【新增】外层循环，遍历不同的基站数量 ---
        for count in gnb_counts_to_test:
            # 内层循环，遍历不同的用户密度
            for density in user_densities_to_test:
                print(f"\n--- 正在测试 num_gnbs = {count}, users_per_gnb = {density} ---")

                # a) 为当前配置创建新的测试环境
                test_config = config.copy()
                test_config['num_gnbs'] = count
                test_config['users_per_gnb'] = density
                test_env = SatTerrestrialEnvironment(test_config)

                # b) 【关键】将主框架的环境临时指向当前测试环境
                dqn_framework.env = test_env

                # c) 初始化当前配置的测试统计变量
                test_episodes = config['testing_episode']
                test_reward_history = []
                total_test_reward, total_test_se, total_qos_violations = 0, 0, 0

                # d) 运行内部测试循环
                for episode in range(test_episodes):
                    state = dqn_framework.env.reset()
                    episode_reward = 0

                    for step in range(config['steps_per_episode']):
                        actions, powers, _ = dqn_framework.select_actions(state, use_exploration=False)
                        next_state, reward, _, qos_violations, spectrum_efficiency = dqn_framework.env.step(actions,
                                                                                                            powers)

                        # 累加各项指标
                        total_test_reward += reward
                        total_test_se += spectrum_efficiency
                        total_qos_violations += qos_violations
                        episode_reward += reward
                        state = next_state

                    avg_episode_reward = episode_reward / config['steps_per_episode']
                    test_reward_history.append(avg_episode_reward)

                # e) 保存当前配置的奖励历史
                REWARD_PATH = save_dir + f"reward_history_{config['model_name']}_{config['mode']}_gnbs_{count}_users_{density}.csv"
                np.savetxt(REWARD_PATH, np.array(test_reward_history), delimiter=",")
                print(f"  测试完成! 结果已保存到 {REWARD_PATH}")

                # f) 打印当前配置的总体平均性能
                total_steps = test_episodes * config['steps_per_episode']
                avg_reward = total_test_reward / total_steps
                avg_se = total_test_se / total_steps
                avg_qos_violations = total_qos_violations / total_steps

                print(f"\n--- DQN (gnbs={count}, users={density}) 整体测试结果 ---")
                print(f"  平均奖励: {avg_reward:.4f}")
                print(f"  平均频谱效率 (bps/Hz): {avg_se:.4f}")
                print(f"  平均每步QoS违规用户数: {avg_qos_violations:.4f}")

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
        # policy = RandomPolicy(env)

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