import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd


# result = [
#     {'Noise Std': 0.00, 'Avg. Reward': 4.6887, 'Avg. Spectrum Efficiency': 36.5527, 'Avg. qos_violations': 2.2664},
#     {'Noise Std': 0.10, 'Avg. Reward': 4.6361, 'Avg. Spectrum Efficiency': 36.4290, 'Avg. qos_violations': 2.3268},
#     {'Noise Std': 0.30, 'Avg. Reward': 4.3661, 'Avg. Spectrum Efficiency': 35.9672, 'Avg. qos_violations': 2.6960},
#     {'Noise Std': 0.50, 'Avg. Reward': 4.2964, 'Avg. Spectrum Efficiency': 35.8227, 'Avg. qos_violations': 2.7772},
#     {'Noise Std': 0.70, 'Avg. Reward': 4.4211, 'Avg. Spectrum Efficiency': 35.8735, 'Avg. qos_violations': 2.5486},
#     {'Noise Std': 1.00, 'Avg. Reward': 4.1492, 'Avg. Spectrum Efficiency': 35.4189, 'Avg. qos_violations': 2.9104}
# ]


# --- 指定字体文件的绝对路径 ---
font_path = '/home/ywx/fonts/TIMES.TTF'  # 请确保这是您服务器上正确的路径和文件名
# ---------------------------------

# 创建一个字体属性对象
try:
    my_font = fm.FontProperties(fname=font_path)
    print(f"成功从路径加载字体: {my_font.get_name()}")
except RuntimeError:
    print(f"错误: 无法从路径 {font_path} 加载字体。请检查路径和文件是否正确。")
    # 如果加载失败，则使用默认字体
    my_font = fm.FontProperties()




# 1. 您的原始数据
data = [
    {'Noise Std': 0.00, 'Avg. Reward': 4.6251, 'Avg. Spectrum Efficiency': 36.3467, 'Avg. qos_violations': 2.3204},
    {'Noise Std': 0.10, 'Avg. Reward': 4.5970, 'Avg. Spectrum Efficiency': 36.3184, 'Avg. qos_violations': 2.3500},
    {'Noise Std': 0.20, 'Avg. Reward': 4.5171, 'Avg. Spectrum Efficiency': 36.3516, 'Avg. qos_violations': 2.5520},
    {'Noise Std': 0.30, 'Avg. Reward': 4.4948, 'Avg. Spectrum Efficiency': 36.0945, 'Avg. qos_violations': 2.5168},
    {'Noise Std': 0.40, 'Avg. Reward': 4.2866, 'Avg. Spectrum Efficiency': 35.6354, 'Avg. qos_violations': 2.8764},
    {'Noise Std': 0.50, 'Avg. Reward': 4.2174, 'Avg. Spectrum Efficiency': 35.8381, 'Avg. qos_violations': 2.9792}
    # {'Noise Std': 0.60, 'Avg. Reward': 4.1747, 'Avg. Spectrum Efficiency': 35.4236, 'Avg. qos_violations': 2.8808},
    # {'Noise Std': 0.70, 'Avg. Reward': 4.3593, 'Avg. Spectrum Efficiency': 35.7162, 'Avg. qos_violations': 2.6128},
    # {'Noise Std': 0.80, 'Avg. Reward': 4.3833, 'Avg. Spectrum Efficiency': 35.7054, 'Avg. qos_violations': 2.7396},
    # {'Noise Std': 0.90, 'Avg. Reward': 4.2302, 'Avg. Spectrum Efficiency': 35.7576, 'Avg. qos_violations': 2.8844},
    # {'Noise Std': 1.00, 'Avg. Reward': 4.5522, 'Avg. Spectrum Efficiency': 35.9410, 'Avg. qos_violations': 2.3336}
]

# 2. 将数据转换为Pandas DataFrame，方便处理
df = pd.DataFrame(data)

# 3. 创建一个包含3个子图的图表
# 3行1列，共享x轴，设置一个合适的画布大小
fig, axes = plt.subplots(3, 1, figsize=(6, 8), sharex=True)

# # 设置图表的主标题
# fig.suptitle('Performance Metrics vs. Noise Std', fontsize=14)

# --- 子图1: Avg. Reward ---
axes[0].plot(df['Noise Std'], df['Avg. Reward'], marker='o', linestyle='-', color='tab:blue')
# axes[0].set_title('Average Reward vs. Noise Std')
axes[0].set_ylabel('Avg. Reward', fontproperties=my_font, fontsize=14)
axes[0].grid(True, linestyle='--', alpha=0.6)

# --- 子图2: Avg. Spectrum Efficiency ---
axes[1].plot(df['Noise Std'], df['Avg. Spectrum Efficiency'], marker='s', linestyle='-', color='tab:green')
# axes[1].set_title('Average Spectrum Efficiency vs. Noise Std')
axes[1].set_ylabel('Avg. Spectrum Efficiency', fontproperties=my_font, fontsize=14)
axes[1].grid(True, linestyle='--', alpha=0.6)

# --- 子图3: Avg. qos_violations ---
axes[2].plot(df['Noise Std'], df['Avg. qos_violations'], marker='^', linestyle='-', color='tab:red')
# axes[2].set_title('Average QoS Violations vs. Noise Std')
axes[2].set_ylabel('Avg. Qos Violations', fontproperties=my_font, fontsize=14)
axes[2].set_xlabel('Noise Std',fontproperties=my_font, fontsize=14) # 只在最下方的图表显示x轴标签
axes[2].grid(True, linestyle='--', alpha=0.6)

# 4. 自动调整子图布局，防止标题和标签重叠
plt.tight_layout(rect=[0, 0, 1, 0.96]) # rect参数为主标题留出空间

# 存储图像
plt.savefig('./sensitive.png')

# 5. 显示图表
plt.show()






