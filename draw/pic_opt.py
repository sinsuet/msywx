import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import itertools

# --- 1. 字体设置 (保持不变) ---
font_path = '/home/ywx/fonts/TIMES.TTF'  # 请确保这是您服务器上正确的路径
try:
    # 尝试加载指定字体
    font_properties = fm.FontProperties(fname=font_path)
    font_properties.set_size(14)
    print(f"成功从路径加载字体: {font_path}")
except Exception as e:
    # 如果失败，回退到默认字体，避免程序崩溃
    print(f"加载字体失败，使用默认字体。错误: {e}")
    font_properties = fm.FontProperties()
    font_properties.set_size(14)


def plot_reward_history(my_font, data_dict, window_size=50, save_path='../result/Convergence analysis/convergence.png'):
    """
    读取并可视化训练过程中的奖励历史。

    参数:
    my_font (FontProperties): 字体属性对象。
    data_dict (dict): 键为图例标签(Label)，值为CSV文件路径(Path)的字典。
                      例如: {'PE-MADDPG': 'path/to/marl.csv', 'IDDQN': 'path/to/dqn.csv'}
    window_size (int): 用于计算移动平均的窗口大小。
    save_path (str): 图片保存路径。
    """

    # 创建图表
    plt.figure(figsize=(10, 6))

    # 定义一个颜色迭代器，确保曲线较多时颜色自动循环
    # 使用 tab10 色板，它包含10种高对比度颜色
    colors = itertools.cycle(plt.cm.tab10.colors)

    # 标记是否有任何数据被成功绘制
    plot_success = False

    # --- 2. 循环处理字典中的每一个模型 ---
    for label_name, file_path in data_dict.items():
        if not os.path.exists(file_path):
            print(f"[警告] 文件不存在，跳过: {file_path}")
            continue

        try:
            # 读取CSV
            df = pd.read_csv(file_path, header=None)

            # 假设奖励数据在第0列
            if df.empty:
                print(f"[警告] 文件为空，跳过: {file_path}")
                continue

            rewards = df[0]

            # 计算移动平均
            if len(rewards) >= window_size:
                moving_avg = rewards.rolling(window=window_size, min_periods=min(10, window_size)).mean()

                # 获取下一个颜色
                current_color = next(colors)

                # 绘制曲线
                plt.plot(moving_avg,
                         label=label_name,
                         color=current_color,
                         linewidth=2,
                         alpha=0.9)  # 稍微加一点透明度防止完全遮挡

                print(f"[成功] 已绘制: {label_name}")
                plot_success = True
            else:
                print(f"[警告] 数据点不足 {window_size} 个，无法计算移动平均: {label_name}")

        except Exception as e:
            print(f"[错误] 处理 {label_name} 时发生错误: {e}")

    # --- 3. 图表美化与保存 ---
    if plot_success:
        plt.xlabel('Episodes', fontsize=14, fontproperties=my_font)
        plt.ylabel('Reward', fontsize=14, fontproperties=my_font)
        plt.grid(True, linestyle='--', alpha=0.6)

        # 图例设置
        plt.legend(fontsize=14, prop=my_font, loc='lower right')
        plt.tight_layout()

        # 确保保存目录存在
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"创建目录: {save_dir}")

        plt.savefig(save_path, dpi=300)  # 增加dpi使图片更清晰
        print(f"图片已保存至: {save_path}")

        # 显示图表 (如果在服务器无头模式下可能需要注释掉)
        # plt.show()
    else:
        print("没有有效的数据被绘制，请检查文件路径或内容。")


if __name__ == "__main__":
    # --- 配置区域 ---

    # 只要维护这个字典，就可以随意增加、删除、修改曲线
    # Key: 图例上显示的名字
    # Value: CSV文件的路径
    models_to_plot = {
        'PE-MADDPG': '../draw/reward_history_marl_train.csv',
        'IDDQN': '../draw/reward_history_dqn_train.csv',
        'MAAC_v0': '../draw/reward_history_marl_train2.csv',  # 假设这是你的Attention模型
        'MAAC_residual_connection': '../draw/reward_history_marl_train3.csv',  # 假设这是你的Attention模型
        'QMIX':      '../draw/reward_history_qmix_train.csv', # 如果想加QMIX，取消注释即可
        # 'New-Model': '../path/to/new_model.csv'             # 随时添加新模型
    }

    # 调用绘图函数
    plot_reward_history(
        my_font=font_properties,
        data_dict=models_to_plot,
        window_size=50,
        save_path='../result/Convergence analysis/convergence_comparison.png'
    )