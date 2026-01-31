import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm

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


# --- 1. 从您提供的图片中提取数据 ---
users_per_gnb = [4, 8, 12, 16]
MARL_values = [22.8006,27.6791,32.0192,37.1380 ]
DQN_values = [24.5239,27.1896,29.6676,33.8397]
GDFP = [24.035,27.0577,27.0577,33.6295]
QMIX = [24.6112, 26.4152, 27.8315, 33.161]
Randompolicy = [34.0185, 33.8247, 34.0363, 33.9969]

# MARL_values = [25.038, 25.8969, 30.9815, 36.4257]
# DQN_values = [20.226, 23.5364, 29.5278, 34.0076]
# GDFP = [25.7368, 26.278, 29.7006, 32.3337]
# QMIX = [24.035, 27.0577, 27.0577, 33.6295]

# --- 2. 设置绘图风格和全局字体 (如果需要) ---
plt.rcParams['font.size'] = 14

# --- 3. 创建画布和第一个Y轴 ---
fig, ax1 = plt.subplots(figsize=(8, 6), dpi=300)

# --- 4. 在第一个Y轴 (ax1) 上绘制RMSE和MAE ---
# 为线条和标签设置颜色
color1 = 'tab:blue'
color2 = 'tab:green'
color3 = 'tab:red'
color4 = 'tab:purple'
color5 = 'tab:orange'
ax1.set_xlabel('Number of users per gNB', fontproperties=my_font, fontsize=16)
ax1.set_ylabel('Average Spectrum Efficiency(bps/Hz)', fontproperties=my_font, fontsize=16)

# --- 核心修改：更新Y轴标签 ---
# -----------------------------
line1, = ax1.plot(users_per_gnb, MARL_values, color=color1, marker='o', linestyle='-', label='PE-MADDPG')
line2, = ax1.plot(users_per_gnb, DQN_values, color=color2, marker='s', linestyle='-', label='IDDQN')
line3, = ax1.plot(users_per_gnb, QMIX, color=color3, marker='o', linestyle='-', label='QMIX')
line4, = ax1.plot(users_per_gnb, GDFP, color=color4, marker='s', linestyle='-', label='GD-FP')
# line5, = ax1.plot(users_per_gnb, Randompolicy, color=color5, marker='s', linestyle='-', label='Randompolicy')



ax1.tick_params(axis='y', labelcolor='black')
ax1.grid(True, which="both", ls="--", linewidth=0.5)

# 手动设置X轴刻度以确保所有点都显示
ax1.set_xticks(users_per_gnb)
ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())



# --- 7. 创建一个统一的图例 ---
# 将三条线的句柄(handles)和标签(labels)合并
lines = [line1, line2, line3, line4]
labels = [l.get_label() for l in lines]
# 为图例也设置字体
legend = ax1.legend(lines, labels, loc='upper left')
for text in legend.get_texts():
    text.set_fontproperties(my_font)
    text.set_fontsize(16)


# --- 8. 添加标题并优化布局 ---
# plt.title('Impact of Rank on Model Performance', fontproperties=my_font, fontsize=16, weight='bold')
fig.tight_layout()  # 自动调整布局以防止标签重叠

# --- 9. 保存并显示图像 ---
plt.savefig('../SE vs user densities/SE vs user densities.png')
plt.show()
