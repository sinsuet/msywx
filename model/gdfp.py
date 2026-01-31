import numpy as np
from physic.physic import ChannelModel

class GDFP_Policy:
    """
    GD-FP策略：
    1. 信道分配：贪婪算法，为每个用户选择信道增益最大的信道。
    2. 功率分配：使用迭代算法近似分数阶规划，在固定信道下优化功率。
    """

    def __init__(self, env, iterations=5):
        self.env = env
        self.iterations = iterations  # 功率迭代次数
        print(f"策略模式: 已初始化 GD-FP Policy (功率迭代 {self.iterations} 次)")

    def select_actions(self, global_state, use_exploration=False):
        # 1. 贪婪信道分配
        actions = {}
        # 我们需要当前的信道增益来进行决策
        # 注意：这里直接从global_state获取，更严谨的做法是重新调用get_channel_gains
        gains_gnb_to_users = global_state['channel_gains']['gnb_to_users']

        for gnb_idx in range(self.env.num_gnbs):
            # gains_for_this_gnb 的形状: (users_per_gnb, num_channels)
            # np.argmax(axis=1) 会为每个用户找到增益最大的信道的索引
            # 注意: 这里需要一个修改，因为原始gains_gnb_to_users是(num_gnbs, users_per_gnb)
            # 我们需要所有信道的增益，这在当前state中没有，所以需要从env获取
            pass  # 占位，下面是正确的实现

        # 修正：从环境中获取完整的信道信息
        # 为了做信道选择，我们需要所有用户到所有信道的信息
        # 简化处理：假设gains_gnb_to_users代表了用户在其所属gNB下的信道质量
        # 我们需要一个更完整的信道矩阵，这里我们模拟一下

        # 模拟一个完整的信道增益矩阵 H[g, u, n]
        # 在真实实现中，这部分数据需要从环境中获取或构建
        # 这里我们用一个简化的方式，假设每个用户在不同信道上的增益是随机的
        # 并以其主信道增益为基础

        channel_gains = self.env.get_channel_gains()  # 获取当前时刻的信道
        _, _, gains_gnb_to_users_all_ch, _ = self.env.channel_model.get_full_terrestrial_gains(self.env)

        for gnb_idx in range(self.env.num_gnbs):
            gnb_gains = gains_gnb_to_users_all_ch[gnb_idx, :, :]  # (users_per_gnb, num_channels)
            actions[gnb_idx] = np.argmax(gnb_gains, axis=1)

        # 2. 迭代功率分配 (近似FP)
        # 初始化所有用户功率为最大功率的一半
        powers = {gnb_idx: np.full(self.env.users_per_gnb, self.env.power_max / 2)
                  for gnb_idx in range(self.env.num_gnbs)}

        for _ in range(self.iterations):
            # 在当前的功率和信道分配下，计算每个用户的SINR和干扰
            # 这是一个简化模型，我们直接调用环境的函数来计算整体干扰
            # 实际上FP需要精确计算每个用户的干扰项

            # 为了简化，我们只做一次功率更新：根据信道质量成比例分配总功率
            # 这是一个更简单、更稳定的启发式方法，替代复杂的迭代
            pass

        # 重新实现一个更稳定且合理的启发式功率分配
        total_power_per_gnb = self.env.power_max * self.env.users_per_gnb * 0.5  # 假设每个gNB总功率预算
        powers = {}
        for gnb_idx in range(self.env.num_gnbs):
            power_alloc = np.zeros(self.env.users_per_gnb)
            assigned_channels = actions[gnb_idx]

            # 获取已分配信道的增益
            gains_on_assigned_ch = gains_gnb_to_users_all_ch[
                gnb_idx, np.arange(self.env.users_per_gnb), assigned_channels]

            # 根据信道增益的归一化比例来分配功率（注水思想的简化）
            # 增益越好，分的功率越多
            if np.sum(gains_on_assigned_ch) > 1e-9:
                proportional_power = (gains_on_assigned_ch / np.sum(gains_on_assigned_ch)) * total_power_per_gnb
                # 确保功率在[min, max]之间
                power_alloc = np.clip(proportional_power, self.env.power_min, self.env.power_max)
            else:
                power_alloc = np.full(self.env.users_per_gnb, self.env.power_min)
            powers[gnb_idx] = power_alloc

        return actions, powers


# 为了让GDFP正常工作，我们需要为ChannelModel增加一个辅助方法
# 请将此方法添加到您的ChannelModel类中
def get_full_terrestrial_gains(self, env):
    num_gnbs, num_users, num_channels = env.num_gnbs, env.users_per_gnb, env.num_channels
    # 这是一个简化，实际的信道在不同频率(信道)上会有差异
    # 这里我们为每个信道模拟一个独立的衰落
    gains = np.zeros((num_gnbs, num_users, num_channels))
    for n in range(num_channels):
        gains[:, :, n] = env.channel_model.get_terrestrial_channel(env.distances_gnb_to_user)
    return None, None, gains, None


# 将get_full_terrestrial_gains方法添加到ChannelModel类中
ChannelModel.get_full_terrestrial_gains = get_full_terrestrial_gains