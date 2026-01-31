# -*- coding: utf-8 -*-
# 最终版本: 包含PRB利用率预测、训练/测试模式切换、模型保存/加载、性能记录

import numpy as np
import time
import datetime

from env.myenv import SatTerrestrialEnvironment

from model.mymodel260129 import SatTerrestrialHPGAMARLFramework
from model.mymodel_graph_2601230 import SatTerrestrialHPGAMARLFramework
from model.dqn import DQN_Framework
from model.gdfp import GDFP_Policy
from model.randompolicy import RandomPolicy
from model.mappo_lagrangian import MAPPOLagrangianFramework

import sys
from trainer import OnPolicyTrainer, OffPolicyTrainer, HeuristicTrainer


# main.py

def experiment_logger(phase, config, description=None, results=None, error=None):
    """
    极简日志记录
    phase: 'start' (开始), 'end' (结束), 'error' (出错)
    """
    import os
    import datetime

    # 路径处理
    path = config.get('LOG_PATH', 'log/experiment_record.txt')
    os.makedirs(os.path.dirname(path), exist_ok=True)

    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    with open(path, 'a', encoding='utf-8') as f:
        if phase == 'start':
            f.write(f"\n[{now}] === 实验开始 ===\n")
            f.write(f"备注: {description}\n")
            f.write(f"模型: {config['model_name']} | 模式: {config['mode']}\n")
            f.write("配置摘要:\n")
            # 只记录关键参数，避免日志太长
            skip_keys = ['model_params', 'mode', 'LOG_PATH']
            for k, v in config.items():
                if k not in skip_keys:
                    f.write(f"  {k}: {v}\n")
            # 记录该模型的特定参数
            if 'model_params' in config:
                f.write(f"  特定参数: {config['model_params'].get(config['model_name'], {})}\n")

        elif phase == 'end':
            f.write(f"[{now}] 实验结束\n")
            f.write(f"结果: {results}\n")
            f.write("-" * 50 + "\n")  # 一个简单的分割线，区分下一次实验

        elif phase == 'error':
            f.write(f"[{now}] 实验异常中断\n")
            f.write(f"错误信息: {error}\n")
            f.write("-" * 50 + "\n")


if __name__ == "__main__":
    config = {
        'mode': 'train',  # 'train' 或 'test'
        'model_name':'marl',  # marl,gdfp,randompolicy,dqn,mappo_lagrangian
        #实验日志
        'LOG_PATH': 'log/experiment_log2.txt',
        # 环境参数
        'num_gnbs': 19, # 建议最大值7或者19
        'users_per_gnb': 16,  # 建议最大值为16
        'num_channels': 10,  # 不改，定义为10。20MHz是一个非常标准的信道带宽
        'terrestrial_qos_mbps': 10,
        'satellite_qos_mbps': 20,
        # 训练通用参数
        'buffer_size': 500000,
        'gamma': 0.99,
        'batch_size': 256,
        'prediction_noise_std': 0.0,
        'reward_scale_factor': 0.5,
        # 收敛性日志参数
        'convergence_window': 100,
        'convergence_delta': 0.01,
        'convergence_std': 0.05,
        'convergence_patience': 3,
        'convergence_log_every': 10,


        'trainging_episode': 200,
        'testing_episode':200,
        'steps_per_episode':25,
        # 训练全尺寸模型
        'max_num_gnbs': 19,
        'max_users_per_gnb': 16,  # 模型和观测向量始终基于这个最大尺寸构建

    'model_params': {
        'marl':{
            'critic_hidden_dim': 512,
            'actor_hidden_dim': 256,
            'critic_lr': 1e-4,
            'actor_lr': 2e-4,
            'save_dir': 'checkpoints/marl/'
        },
        'dqn': {
            # 'critic_hidden_dim': 256,
            # 'actor_hidden_dim': 128,
            # 'critic_lr': 1e-4,
            # 'actor_lr': 3e-4,
            # 'num_power_levels': 5,
            'dqn_hidden_dim': 128,  # DQN 网络的隐藏层维度
            'dqn_lr': 3e-4,  # DQN 网络的学习率
            'num_power_levels': 5,  # 功率离散化等级
            'epsilon_start': 0.95,  # 探索率初始值
            'epsilon_decay': 0.9995,  # 探索率衰减因子
            'epsilon_min': 0.05,  # 探索率最小值
            'target_update_freq': 100,  # Target网络同步的频率（按step计算）
            'save_dir':'checkpoints/dqn/'
            # 'dqn_model_path': "checkpoints/dqn/best_dqn_model.pth",
            # 'reward_history_path': "checkpoints/dqn/reward_history_dqn.csv",
        },
        'gdfp': {
            'save_dir': 'checkpoints/gdfp/'
        },
        'randompolicy':{
            'save_dir': 'checkpoints/randompolicy/'
        },
        'mappo_lagrangian': {
            # 网络结构参数
            'actor_hidden_dim': 128,
            'critic_hidden_dim': 256,
            'aux_output_dim': 7,  # 对应 max_gnbs，预测每个基站的负载/拥塞

            # 学习率参数 (降低学习率以稳定训练)
            'actor_lr': 1e-4,
            'critic_lr': 1e-4,
            'dual_lr': 5e-3,  # 拉格朗日乘子的学习率 (对偶梯度上升步长)

            # PPO 算法参数
            'gae_lambda': 0.95,  # GAE 优势估计的平滑因子
            'clip_ratio': 0.1,  # PPO 裁剪范围 (0.1 ~ 0.3)，更保守的裁剪
            'entropy_coef': 0.05,  # 熵正则化系数，提高以鼓励更多探索

            # 关键机制参数
            'aux_coef': 0.1,  # 辅助预测任务的损失权重
            'cost_limit': 0.50,  # QoS 违规率的硬约束上限 (放宽到 50%)

            # 路径
            'save_dir': 'checkpoints/mappo_lagrangian/'

        }

    }

    }


def get_trainer(config):
    """工厂函数：根据配置返回对应的 Trainer 和 Framework"""
    name = config['model_name']

    # 1. 实例化 Framework (模型主体)
    if name == 'mappo_lagrangian':
        framework = MAPPOLagrangianFramework(config)
        trainer_cls = OnPolicyTrainer

    elif name == 'dqn':
        framework = DQN_Framework(config)
        trainer_cls = OffPolicyTrainer

    elif name == 'marl':
        framework = SatTerrestrialHPGAMARLFramework(config)
        trainer_cls = OffPolicyTrainer

    elif name == 'gdfp':
        from env.myenv import SatTerrestrialEnvironment
        env = SatTerrestrialEnvironment(config)
        framework = GDFP_Policy(env)
        framework.env = env  # 注入环境以便 Trainer 使用
        trainer_cls = HeuristicTrainer

    elif name == 'randompolicy':
        from env.myenv import SatTerrestrialEnvironment
        env = SatTerrestrialEnvironment(config)
        framework = RandomPolicy(env)
        framework.env = env
        trainer_cls = HeuristicTrainer

    else:
        raise ValueError(f"Unknown model name: {name}")

    # 2. 实例化 Trainer
    return trainer_cls(framework, config)


if __name__ == "__main__":
    # 获取 Trainer
    try:
        note = input("本次实验备注 (回车跳过): ").strip() or "常规运行"
    except EOFError:
        note = "常规运行"

    # 记录开始
    experiment_logger('start', config, description=note)

    try:
        trainer = get_trainer(config)

        # 2. 执行核心逻辑
        if config['mode'] == 'train':
            trainer.train()
            # 获取结果 (根据你的trainer变量名自行调整)
            res = {
                "Best_Reward": getattr(trainer, 'best_avg_reward', None),
                "Final_Cost": getattr(trainer, 'cost_history', [])[-1] if hasattr(trainer,
                                                                                  'cost_history') and trainer.cost_history else None
            }
        else:
            trainer.test()
            res = "Test Finished"

        # 3. 记录正常结束
        experiment_logger('end', config, results=res)
        print("实验完成，日志已记录。")

    except Exception as e:
        # 4. 记录报错，这对科研排错很重要
        import traceback

        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"程序出错: {e}")
        experiment_logger('error', config, error=error_msg)
        raise e  # 继续抛出异常，让IDE能停在报错行