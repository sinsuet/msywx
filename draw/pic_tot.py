import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm




font_path = '/home/ywx/fonts/TIMES.TTF'  # 请确保这是您服务器上正确的路径和文件名
try:
    font_properties = fm.FontProperties(fname=font_path)
    font_properties.set_size(14)  # 直接修改 my_font 对象的字号
    print(f"成功从路径加载字体: {font_properties.get_name()}")
except Exception as e:
    font_properties = fm.FontProperties()


def plot_reward_history(my_font, marl, iddqn, qmix, window_size=50):
    """
    读取并可视化训练过程中的奖励历史。

    参数:
    csv_path (str): reward_history.csv文件的路径。
    window_size (int): 用于计算移动平均的窗口大小。
    """

    # 使用pandas读取CSV文件，header=None表示文件没有标题行
    marl = pd.read_csv(marl, header=None)
    iddqn = pd.read_csv(iddqn, header=None)
    qmix = pd.read_csv(qmix, header=None)
    rewards_marl = marl[0]
    rewards_iddqn = iddqn[0]
    rewards_qmix = qmix[0]



    # 创建图表
    plt.figure(figsize=(10, 6))

    # 绘制原始的逐回合平均奖励
    # plt.plot(rewards_marl, label='Reward', color='lightblue', alpha=0.6)

    # 计算并绘制移动平均奖励，以显示趋势
    if len(rewards_marl) >= window_size:
        # moving_avg = np.convolve(rewards_marl, np.ones(window_size)/window_size, mode='valid')
        # 计算移动平均值 (关键逻辑)
        moving_avg_marl = rewards_marl.rolling(window=window_size, min_periods=10).mean()
        moving_avg_iddqn = rewards_iddqn.rolling(window=window_size, min_periods=10).mean()
        moving_avg_qmix = rewards_qmix.rolling(window=window_size, min_periods=10).mean()

        plt.plot(moving_avg_marl, label=f'PE-MADDPG', color='tab:blue', linewidth=2)
        plt.plot(moving_avg_iddqn, label=f'IDDQN', color='tab:purple', linewidth=2)
        plt.plot(moving_avg_qmix, label=f'QMIX', color='tab:orange', linewidth=2)

        # 对齐x轴
        # plt.plot(np.arange(window_size - 1, len(rewards_marl)), moving_avg, label=f'Average Reward (window size={window_size})', color='orange', linewidth=2)

    # 设置图表标题和标签
    # plt.title('MADDPG训练过程中的奖励变化趋势', fontsize=16)
    plt.xlabel('Episodes', fontsize=14, fontproperties=my_font)
    plt.ylabel('Reward', fontsize=14, font_properties=my_font)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=14, prop=my_font, loc = 'lower right')
    plt.tight_layout()

    plt.savefig('../result/Convergence analysis/convergence.png')

    # 显示图表
    plt.show()

if __name__ == "__main__":
    # 您可以将CSV文件的路径作为参数传递给函数
    # 例如: plot_reward_history("path/to/your/reward_history.csv")
    dqn_reward_path = '../draw/reward_history_dqn_train.csv'
    marl_reward_path = '../draw/reward_history_marl_train.csv'
    marl_reward_path_attention = '../draw/reward_history_marl_train2.csv'
    qmix_reward_path = '../draw/reward_history_qmix_train.csv'

    plot_reward_history(font_properties,marl_reward_path,dqn_reward_path,marl_reward_path_attention)
