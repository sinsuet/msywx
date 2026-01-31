import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm




font_path = '/home/ywx/fonts/TIMES.TTF'  # 请确保这是您服务器上正确的路径和文件名
try:
    font_properties = fm.FontProperties(fname=font_path)
    print(f"成功从路径加载字体: {font_properties.get_name()}")
except Exception as e:
    print(f"错误: 无法从路径 {font_path} 加载字体。将使用默认字体。")
    font_properties = fm.FontProperties()


def plot_reward_history(my_font, csv_path="reward_history.csv", window_size=50):
    """
    读取并可视化训练过程中的奖励历史。

    参数:
    csv_path (str): reward_history.csv文件的路径。
    window_size (int): 用于计算移动平均的窗口大小。
    """
    try:
        # 使用pandas读取CSV文件，header=None表示文件没有标题行
        data = pd.read_csv(csv_path, header=None)
        rewards = data[0]
        print(f"成功从'{csv_path}'中读取了 {len(rewards)} 个奖励记录。")
    except FileNotFoundError:
        print(f"错误: 未找到文件 '{csv_path}'。")
        print("请确保该文件与本脚本在同一目录下，或者提供正确的文件路径。")
        return
    except pd.errors.EmptyDataError:
        print(f"错误: 文件 '{csv_path}' 为空。")
        return

    # 创建图表
    plt.figure(figsize=(12, 6))

    # 绘制原始的逐回合平均奖励
    plt.plot(rewards, label='Reward', color='lightblue', alpha=0.6)

    # 计算并绘制移动平均奖励，以显示趋势
    if len(rewards) >= window_size:
        # moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        # 计算移动平均值 (关键逻辑)
        moving_avg = rewards.rolling(window=window_size, min_periods=10).mean()
        plt.plot(moving_avg, label=f'Moving Average (window={window_size})', color='orange', linewidth=2.5)

        # 对齐x轴
        # plt.plot(np.arange(window_size - 1, len(rewards)), moving_avg, label=f'Average Reward (window size={window_size})', color='orange', linewidth=2)

    # 设置图表标题和标签
    # plt.title('MADDPG训练过程中的奖励变化趋势', fontsize=16)
    plt.xlabel('Episodes', fontsize=14, fontproperties=my_font)
    plt.ylabel('Reward', fontsize=14, font_properties=my_font)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=14,prop=my_font)
    plt.tight_layout()

    # 显示图表
    plt.show()

if __name__ == "__main__":
    # 您可以将CSV文件的路径作为参数传递给函数
    # 例如: plot_reward_history("path/to/your/reward_history.csv")
    dqn_reward_path = '../checkpoints/dqn/reward_history_dqn_train.csv'
    marl_reward_path = '../checkpoints/marl/reward_history_marl_train.csv'
    qmix_reward_path = '../checkpoints/QMIX/reward_history_qmix_train.csv'

    plot_reward_history(font_properties,marl_reward_path)
