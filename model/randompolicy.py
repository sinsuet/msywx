import numpy as np

class RandomPolicy:
    """
    随机策略：为每个用户的信道和功率进行随机分配。
    """

    def __init__(self, env):
        self.env = env

    def select_actions(self, global_state, use_exploration=False):
        """
        为所有gNB的所有用户随机选择动作。
        'global_state' 和 'use_exploration' 参数在此处未使用，仅为保持接口一致。
        """
        actions = {}
        powers = {}
        for gnb_idx in range(self.env.num_gnbs):
            # 为每个用户随机选择一个信道
            channel_actions = np.random.randint(0, self.env.num_channels, self.env.users_per_gnb)

            # 为每个用户在允许范围内随机选择一个功率
            power_actions = np.random.uniform(self.env.power_min, self.env.power_max, self.env.users_per_gnb)

            actions[gnb_idx] = channel_actions
            powers[gnb_idx] = power_actions

        return actions, powers

