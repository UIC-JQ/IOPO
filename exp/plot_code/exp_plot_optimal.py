
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------------------------

# format:
# 5_1 -> 5_2 -> 7_1 -> 7_2
train_optimal = [393.7888430199746, 238.84310538482003, 608.7916256932635, 384.2801599603325]
train_final = [393.78887939453125, 250.95254516601562, 609.0957641601562, 389.3298034667969]
train_begin = [413.2532958984375, 400.24530029296875, 736.1500244140625, 481.6894226074219]

# 准备数据
x = np.array([1, 2, 3, 4])  # X轴的位置
y_REF = [1, 1, 1, 1]  # 第一个柱子的高度
y_train_final = [train_optimal[i] / train_final[i] for i in range(4)]  # 第二个柱子的高度
y_train_begin = [train_optimal[i] / train_begin[i] for i in range(4)]  # 第三个柱子的高度

width = 0.25  # 柱子的宽度
# 绘制柱形图
plt.figure(figsize=(10, 7))
plt.bar(x, y_REF, width=width, label='RATIO: OPTIMAL', hatch='', color='lightgrey', edgecolor='black', linewidth=1)
plt.bar(x + width, y_train_final, width=width, label='RATIO: OPTIMAL / TRAIN_FINAL_ENG_COST', color='#ddddff', edgecolor='black', linewidth=1)
plt.bar(x + 2 * width, y_train_begin, width=width, label='RATIO: OPTIMAL / TRAIN_BEGIN_ENG_COST', color='#ffffdd', edgecolor='black', linewidth=1)

dff = 0.035
plt.annotate('', xy=(1.25, 1.00), xytext=(1.5, 0.95 + 0.01), arrowprops=dict(arrowstyle='->', linewidth=1.5))
plt.annotate('', xy=(2.25, 0.95), xytext=(2.5, 0.60 + dff), arrowprops=dict(arrowstyle='->', linewidth=1.5))
plt.annotate('', xy=(3.25, 1.00), xytext=(3.5, 0.83 + dff), arrowprops=dict(arrowstyle='->', linewidth=1.5))
plt.annotate('', xy=(4.25, 0.99), xytext=(4.5, 0.80 + dff), arrowprops=dict(arrowstyle='->', linewidth=1.5))

# 添加标题和标签
plt.title('Final Ratio On Training Set', fontsize=12, fontweight='bold')
plt.xlabel('Setting Format: (Number of User, Number of UAVs)', fontsize=12, fontweight='bold')
plt.ylabel('Ratio', fontsize=12, fontweight='bold')

xticks = [1.25, 2.25, 3.25, 4.25]  # Y轴刻度标签
labels = ["(5, 1)", "(5, 2)", "(7, 1)", "(7, 2)"]  # 自定义的X轴刻度标签
plt.xticks(xticks, labels)
yticks = [0.2, 0.4, 0.6, 0.8, 1.0]  # Y轴刻度标签
plt.yticks(yticks)
plt.ylim(0, 1.3)  # 设置Y轴的最小值和最大值

for i, v in enumerate(y_train_final):
    plt.text(x[i] + width, v + 0.01, '{:.2f}'.format(v), ha='center', fontsize=13)

for i, v in enumerate(y_train_begin):
    plt.text(x[i] + 2*width, v + 0.01, '{:.2f}'.format(v), ha='center', fontsize=13)

# 添加图例
plt.legend()

# 展示图表
plt.savefig('./train_optimal_converge_exp.pdf')

# -------------------------------------------------------------------------
# test exp
# 5_1 -> 5_2 -> 7_1 -> 7_2
test_optimal = [395.9494635129852, 239.04496839020135, 604.2852650769788, 386.73873205263004]
test_final = [395.9494564074575, 253.38024503597404, 606.366816950202, 396.35039221570304]

# 准备数据
x = np.array([1, 2, 3, 4])  # X轴的位置
y_REF = [1, 1, 1, 1]  # 第一个柱子的高度
y_test_final = [test_optimal[i] / test_final[i] for i in range(4)]  # 第二个柱子的高度

width = 0.25  # 柱子的宽度
# 绘制柱形图
plt.figure(figsize=(10, 7))
plt.bar(x, y_REF, width=width, label='RATIO: OPTIMAL', hatch='', color='lightgrey', edgecolor='black', linewidth=1)
plt.bar(x + width, y_test_final, width=width, label='RATIO: OPTIMAL / TEST_FINAL_ENG_COST', color='#ddddff', edgecolor='black', linewidth=1)

dff = 0.035
# plt.annotate('', xy=(1.25, 1.00), xytext=(1.5, 0.95 + 0.01), arrowprops=dict(arrowstyle='->', linewidth=1.5))
# plt.annotate('', xy=(2.25, 0.95), xytext=(2.5, 0.60 + dff), arrowprops=dict(arrowstyle='->', linewidth=1.5))

# 添加标题和标签
plt.title('Final Ratio On Test Set', fontsize=12, fontweight='bold')
plt.xlabel('Setting Format: (Number of User, Number of UAVs)', fontsize=12, fontweight='bold')
plt.ylabel('Ratio', fontsize=12, fontweight='bold')

xticks = [1.125, 2.125, 3.125, 4.125]  # Y轴刻度标签
labels = ["(5, 1)", "(5, 2)", "(7, 1)", "(7, 2)"]  # 自定义的X轴刻度标签
plt.xticks(xticks, labels)
yticks = [0.2, 0.4, 0.6, 0.8, 1.0]  # Y轴刻度标签
plt.yticks(yticks)
plt.ylim(0, 1.3)  # 设置Y轴的最小值和最大值

for i, v in enumerate(y_test_final):
    plt.text(x[i] + width, v + 0.01, '{:.2f}'.format(v), ha='center', fontsize=13)

# 添加图例
plt.legend()

# 展示图表
plt.savefig('./test_optimal_converge_exp.pdf')
