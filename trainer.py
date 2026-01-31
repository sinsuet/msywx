# trainer.py
import time
import numpy as np
import os
import datetime


class BaseTrainer:
    def __init__(self, framework, config):
        self.framework = framework
        self.config = config
        self.mode = config['mode']
        self.model_name = config['model_name']

        # 1. 路径设置
        model_params = config['model_params'][self.model_name]
        self.save_dir = model_params['save_dir']
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.model_path = os.path.join(self.save_dir, f"best_model_{self.model_name}.pth")
        self.reward_path = os.path.join(self.save_dir, f"reward_history_{self.model_name}_{self.mode}.csv")
        self.log_path = os.path.join(self.save_dir, f"experiment_log_{self.model_name}_{self.mode}.txt")

        self.best_avg_reward = -float('inf')
        self.reward_history = []
        self.cost_history = []
        self.convergence_window = int(config.get('convergence_window', 100))
        self.convergence_delta = float(config.get('convergence_delta', 0.01))
        self.convergence_std = float(config.get('convergence_std', 0.05))
        self.convergence_patience = int(config.get('convergence_patience', 3))
        self.convergence_log_every = int(config.get('convergence_log_every', 10))
        self._convergence_streak = 0

    def log(self, message):
        print(message)
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")

    def save_checkpoint(self, avg_reward, episode):
        if avg_reward > self.best_avg_reward and self.mode == 'train':
            self.best_avg_reward = avg_reward
            if hasattr(self.framework, 'save_models'):
                # 传入完整路径，依赖 mappo_lagrangian.py 的修复版或直接传文件名
                try:
                    self.framework.save_models(self.model_path)
                except:
                    # 兼容旧代码可能只接受文件名的情况
                    self.framework.save_models(f"best_model_{self.model_name}.pth")
            self.log(f"*** Ep {episode}: New best reward {self.best_avg_reward:.4f} | Saved ***")

    def save_history(self):
        np.savetxt(self.reward_path, np.array(self.reward_history), delimiter=",")
        self.log(f"History saved to {self.reward_path}")

    def _get_action(self, state, use_exploration):
        """统一不同框架的动作选择接口"""
        # 优先尝试 select_actions
        if hasattr(self.framework, 'select_actions'):
            return self.framework.select_actions(state, use_exploration=use_exploration)
        # 其次尝试 hierarchical_decision (旧MARL)
        elif hasattr(self.framework, 'hierarchical_decision'):
            return self.framework.hierarchical_decision(state, use_exploration=use_exploration)
        else:
            raise NotImplementedError("Framework must implement select_actions or hierarchical_decision")

    def _log_convergence(self, episode, avg_reward, avg_cost=None):
        window = self.convergence_window
        if len(self.reward_history) < window:
            msg = (
                f"[Conv] Ep {episode}: "
                f"insufficient history ({len(self.reward_history)}/{window})"
            )
            if avg_cost is not None:
                msg += f" | AvgC={avg_cost:.3f}"
            self.log(msg)
            return

        curr_window = np.array(self.reward_history[-window:])
        curr_mean = float(np.mean(curr_window))
        curr_std = float(np.std(curr_window))

        delta = None
        if len(self.reward_history) >= 2 * window:
            prev_window = np.array(self.reward_history[-2 * window:-window])
            prev_mean = float(np.mean(prev_window))
            delta = curr_mean - prev_mean

        is_stable = (
            delta is not None
            and abs(delta) < self.convergence_delta
            and curr_std < self.convergence_std
        )
        if is_stable:
            self._convergence_streak += 1
        else:
            self._convergence_streak = 0

        msg = (
            f"[Conv] Ep {episode}: "
            f"win={window} mean={curr_mean:.3f} std={curr_std:.3f}"
        )
        if delta is not None:
            msg += f" | delta={delta:.3f}"
        if avg_cost is not None:
            msg += f" | AvgC={avg_cost:.3f}"
        msg += f" | stable={is_stable} streak={self._convergence_streak}"
        if self._convergence_streak >= self.convergence_patience:
            msg += " | converged=True"
        self.log(msg)

    def train(self):
        raise NotImplementedError

    def test(self):
        self.log(f"\n--- 开始测试: {self.model_name} ---")

        # 加载模型 (Heuristic 不需要)
        if self.model_name not in ['gdfp', 'randompolicy']:
            if hasattr(self.framework, 'load_models'):
                self.framework.load_models(self.model_path)
            # 设置评估模式
            if hasattr(self.framework, 'agents'):
                # 兼容 DQN 和 MARL
                agents_list = self.framework.agents
                for agent in agents_list:
                    if hasattr(agent, 'actor'): agent.actor.eval()  # MARL
                    if hasattr(agent, 'q_network'): agent.q_network.eval()  # DQN
                    if hasattr(agent, 'eval'): agent.eval()  # MAPPO

        test_rewards = []
        test_se = []
        test_qos = []

        test_episodes = self.config['testing_episode']

        for episode in range(test_episodes):
            state = self.framework.env.reset()
            ep_reward = 0
            ep_se = 0
            ep_qos = 0

            for step in range(self.config['steps_per_episode']):
                # 获取动作 (关闭探索)
                res = self._get_action(state, use_exploration=False)
                # 兼容返回 2个值 或 3个值的情况
                actions, powers = res[0], res[1]

                next_state, reward, _, qos_violations, spectrum_efficiency = self.framework.env.step(actions, powers)

                ep_reward += reward
                ep_se += spectrum_efficiency
                ep_qos += qos_violations
                state = next_state

            avg_ep_reward = ep_reward / self.config['steps_per_episode']
            test_rewards.append(avg_ep_reward)
            test_se.append(ep_se / self.config['steps_per_episode'])
            test_qos.append(ep_qos / self.config['steps_per_episode'])

            if (episode + 1) % 50 == 0:
                print(f"Testing... {episode + 1}/{test_episodes}")

        final_res = f"""
        === 测试完成 ===
        平均奖励: {np.mean(test_rewards):.4f}
        平均频谱效率: {np.mean(test_se):.4f}
        平均QoS违规: {np.mean(test_qos):.4f}
        """
        self.log(final_res)
        np.savetxt(self.reward_path, np.array(test_rewards), delimiter=",")


class OnPolicyTrainer(BaseTrainer):
    """
    适用于 MAPPO 等 On-Policy 算法 (回合结束后更新)
    """

    def train(self):
        self.log(f"\n--- 开始 On-Policy 训练: {self.config['model_name']} ---")
        episodes = self.config['trainging_episode']
        steps = self.config['steps_per_episode']

        # 【关键修正】在循环外初始化变量，防止报错
        last_lambda = 0.0
        last_critic_loss = 0.0

        for episode in range(episodes):
            state = self.framework.env.reset()
            ep_reward = 0
            ep_cost = 0

            # 计时统计
            t_stats = {'decision': 0, 'env': 0, 'store': 0, 'train': 0}

            for step in range(steps):
                # 1. 决策
                t0 = time.time()
                actions, powers = self._get_action(state, use_exploration=True)
                t_stats['decision'] += time.time() - t0

                # 2. 环境交互
                t1 = time.time()
                next_state, reward, _, qos_violations, _ = self.framework.env.step(actions, powers)
                t_stats['env'] += time.time() - t1

                ep_reward += reward
                # Cost 只计算地面用户的 QoS 违规率
                current_cost = qos_violations / self.config['users_per_gnb'] / self.config['num_gnbs']
                ep_cost += current_cost

                # 3. 存储
                t2 = time.time()
                done = (step == steps - 1)
                aux_target = next_state.get('prb_prediction', None)

                self.framework.store_transition(reward=reward, cost=current_cost, done=done, aux_target=aux_target)
                t_stats['store'] += time.time() - t2

                state = next_state

            # 4. 训练 (回合结束)
            t3 = time.time()
            train_info = self.framework.train_step()
            t_stats['train'] += time.time() - t3

            # 记录与保存
            avg_reward = ep_reward / steps
            avg_cost = ep_cost / steps
            self.reward_history.append(avg_reward)
            self.cost_history.append(avg_cost)

            self.save_checkpoint(avg_reward, episode)

            # 【逻辑优化】
            # 如果发生了训练 (train_info 非空)，更新 last_lambda
            # 如果没发生训练，保持 last_lambda 不变 (这样日志就不会显示 0.000 而是显示上一次的值)
            if train_info:
                last_lambda = train_info.get('lambda', last_lambda)
                last_critic_loss = train_info.get('loss_critic', last_critic_loss)

            if episode % 10 == 0:
                # 【修正打印】直接使用维护好的 last_变量，这样即使本回合没训练，也能看到当前的 Lambda 值
                self.log(
                    f"Ep {episode}: R={avg_reward:.3f} | C={avg_cost:.3f} | Lam={last_lambda:.3f} | L_Critic={last_critic_loss:.3f}")
            if episode % self.convergence_log_every == 0:
                self._log_convergence(episode, avg_reward, avg_cost=avg_cost)

        self.save_history()


class OffPolicyTrainer(BaseTrainer):
    """
    适用于 DQN, MARL (旧版)
    """

    def train(self):
        self.log(f"\n--- 开始 Off-Policy 训练: {self.model_name} ---")
        episodes = self.config['trainging_episode']
        steps = self.config['steps_per_episode']
        global_step = 0
        start_time = time.time()
        last_critic_loss = None
        last_actor_loss = None
        critic_loss_history = []
        actor_loss_history = []
        se_history = []
        qos_history = []

        # 预先检查是否有 target update 方法
        has_target_update = False
        target_update_freq = 100
        if self.model_name == 'dqn':
            target_update_freq = self.config['model_params']['dqn']['target_update_freq']
            has_target_update = True

        for episode in range(episodes):
            state = self.framework.env.reset()
            ep_reward = 0
            ep_se = 0
            ep_qos = 0

            # 耗时统计
            time_stats = {'decision': 0, 'env': 0, 'push': 0, 'train': 0}

            for step in range(steps):
                global_step += 1

                # 1. 决策
                t0 = time.time()
                res = self._get_action(state, use_exploration=True)
                actions, powers = res[0], res[1]
                time_stats['decision'] += time.time() - t0

                # 2. 交互
                t1 = time.time()
                next_state, reward, _, qos_violations, spectrum_efficiency = self.framework.env.step(actions, powers)
                time_stats['env'] += time.time() - t1
                ep_reward += reward
                ep_se += spectrum_efficiency
                ep_qos += qos_violations

                # 3. 存储 (兼容 DQN 和 MARL 的差异)
                t2 = time.time()
                if self.model_name == 'dqn':
                    # DQN 返回的 res[2] 是 joint_actions
                    joint_actions = res[2]
                    self.framework.replay_buffer.push(state, joint_actions, reward, next_state, False)
                else:
                    # MARL (旧) 需要 combined_actions 字典
                    combined = {g: (actions[g], powers[g]) for g in actions}
                    self.framework.replay_buffer.push(state, combined, reward, next_state, False)
                time_stats['push'] += time.time() - t2

                # 4. 训练
                t3 = time.time()
                if len(self.framework.replay_buffer) > self.config['batch_size']:
                    train_info = self.framework.train_step()
                    if train_info and 'loss_critic' in train_info:
                        last_critic_loss = train_info['loss_critic']
                    if train_info and 'loss_actor' in train_info:
                        last_actor_loss = train_info['loss_actor']
                time_stats['train'] += time.time() - t3

                # 5. Target Update (DQN特有)
                if has_target_update and global_step % target_update_freq == 0:
                    for agent in self.framework.agents:
                        if hasattr(agent, 'update_target_network'):
                            agent.update_target_network()

                state = next_state

            avg_reward = ep_reward / steps
            avg_se = ep_se / steps
            avg_qos = ep_qos / steps
            self.reward_history.append(avg_reward)
            critic_loss_history.append(last_critic_loss)
            actor_loss_history.append(last_actor_loss)
            se_history.append(avg_se)
            qos_history.append(avg_qos)

            self.save_checkpoint(avg_reward, episode)

            # 自适应学习率调度（基于最近100回合均值更稳）
            if hasattr(self.framework, 'step_schedulers'):
                recent_window = 100
                if len(self.reward_history) >= recent_window:
                    recent_mean = float(np.mean(self.reward_history[-recent_window:]))
                    self.framework.step_schedulers(recent_mean)

            if episode % 10 == 0:
                recent_10 = float(np.mean(self.reward_history[-10:])) if len(self.reward_history) >= 10 else avg_reward
                recent_50 = float(np.mean(self.reward_history[-50:])) if len(self.reward_history) >= 50 else avg_reward
                recent_10_se = float(np.mean(se_history[-10:])) if len(se_history) >= 10 else avg_se
                recent_10_qos = float(np.mean(qos_history[-10:])) if len(qos_history) >= 10 else avg_qos
                recent_critics = [v for v in critic_loss_history[-10:] if v is not None]
                recent_actors = [v for v in actor_loss_history[-10:] if v is not None]
                recent_crit_loss = float(np.mean(recent_critics)) if recent_critics else None
                recent_actor_loss = float(np.mean(recent_actors)) if recent_actors else None
                elapsed_min = (time.time() - start_time) / 60.0
                avg_per_ep = elapsed_min / max(1, episode + 1)
                remaining_min = avg_per_ep * (episodes - episode - 1)
                critic_loss_str = f"{recent_crit_loss:.3f}" if recent_crit_loss is not None else "N/A"
                actor_loss_str = f"{recent_actor_loss:.3f}" if recent_actor_loss is not None else "N/A"
                msg = (
                    f"Ep {episode:4d}/{episodes} ({(episode + 1) / episodes * 100:5.1f}%) | "
                    f"Reward={avg_reward:.4f} | Recent-10={recent_10:.4f} | "
                    f"Recent-50={recent_50:.4f} | Best={self.best_avg_reward:.4f} | "
                    f"Critic_Loss={critic_loss_str} | Actor_Loss={actor_loss_str} | "
                    f"SE={avg_se:.4f} (R10={recent_10_se:.4f}) | "
                    f"QoS={avg_qos:.4f} (R10={recent_10_qos:.4f}) | "
                    f"Time={elapsed_min:.1f}m | "
                    f"Remaining={remaining_min:.1f}m"
                )
                self.log(msg)
            if episode % self.convergence_log_every == 0:
                self._log_convergence(episode, avg_reward)

        self.save_history()


class HeuristicTrainer(BaseTrainer):
    """GDFP, Random"""
    def train(self):
        self.log(f"策略 {self.model_name} 不需要训练，直接进入测试模式。")
        self.test()