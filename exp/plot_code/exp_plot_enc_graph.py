import matplotlib.pyplot as plt
import csv
import numpy as np

def read_csv(filename, diff=2500):
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        for line in csv_reader:
            points = [float(v) for v in line]
            break

        ret = []
        for i in range(0, len(points), diff):
            ret.append(np.mean(points[i: i + diff]))

    return ret

x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# 绘制完整的折线图
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(x, y, label='Original Data')

axins = inset_axes(ax, width="30%", height="30%", loc='upper left',
                   bbox_to_anchor=(0.1, 0.1, 1, 1),
                   bbox_transform=ax.transAxes)

# 显示图形
plt.show()

# -------------------------------------------------------------------------------------------
# data_file_10_3 = '5_25_Log/user:10_uav:3/TrainingLoss_MODEL_MLP_NumOfUser:10_NumOf_UAV:3.csv'
# data_file_15_3 = '5_25_Log/user:15_uav:3/TrainingLoss_MODEL_MLP_NumOfUser:15_NumOf_UAV:3.csv'
# data_file_20_3 = '5_25_Log/user:20_uav:3/TrainingLoss_MODEL_MLP_NumOfUser:20_NumOf_UAV:3.csv'
# data_file_20_4 = '5_25_Log/user:20_uav:4/TrainingLoss_MODEL_MLP_NumOfUser:20_NumOf_UAV:4.csv'
# data_file_20_5 = '5_25_Log/user:20_uav:5/TrainingLoss_MODEL_MLP_NumOfUser:20_NumOf_UAV:5.csv'


# df = 25
# l10_3 = read_csv(data_file_10_3, diff=df)
# l15_3 = read_csv(data_file_15_3, diff=df)
# l20_3 = read_csv(data_file_20_3, diff=df)
# l20_4 = read_csv(data_file_20_4, diff=df)
# l20_5 = read_csv(data_file_20_5, diff=df)

# plt.figure(figsize=(10, 5))
# x = np.arange(1, len(l10_3) + 1, step=1)
# plt.plot(x, l10_3, label='Loss: 10 Users, 3 UAVs')
# plt.plot(x, l15_3, label='Loss: 15 Users, 3 UAVs')
# plt.plot(x, l20_3, label='Loss: 20 Users, 3 UAVs')
# plt.plot(x, l20_4, label='Loss: 20 Users, 4 UAVs')
# plt.plot(x, l20_5, label='Loss: 20 Users, 5 UAVs')

# plt.title("Training Loss of DNN", fontsize=14)
# plt.xlabel("Time Frames", fontsize=12)
# plt.ylabel("Loss", fontsize=12)
# plt.grid(True, alpha=0.5)
# plt.legend()
# plt.savefig('./Training_Loss.pdf')
# # -----------------------------------------------------------------

# eng_file_10_3 = '5_25_Log/user:10_uav:3/EngCostDuringTraining_TI:10_MemS:384.csv'
# eng_file_15_3 = '5_25_Log/user:15_uav:3/EngCostDuringTraining_TI:10_MemS:384.csv'
# eng_file_20_3 = '5_25_Log/user:20_uav:3/EngCostDuringTraining_TI:10_MemS:384.csv'
# eng_file_20_4 = '5_25_Log/user:20_uav:4/EngCostDuringTraining_TI:10_MemS:384.csv'
# eng_file_20_5 = '5_25_Log/user:20_uav:5/EngCostDuringTraining_TI:10_MemS:384.csv'
# e10_3 = read_csv(eng_file_10_3, diff=df*10)
# e15_3 = read_csv(eng_file_15_3, diff=df*10)
# e20_3 = read_csv(eng_file_20_3, diff=df*10)
# e20_4 = read_csv(eng_file_20_4, diff=df*10)
# e20_5 = read_csv(eng_file_20_5, diff=df*10)
# plt.figure(figsize=(10, 5))
# x = np.arange(1, len(e10_3) + 1, step=1)

# plt.plot(x, e10_3, label='10 Users, 3 UAVs')
# plt.plot(x, e15_3, label='15 Users, 3 UAVs')
# plt.plot(x, e20_3, label='20 Users, 3 UAVs')
# plt.plot(x, e20_4, label='20 Users, 4 UAVs')
# plt.plot(x, e20_5, label='20 Users, 5 UAVs')

# plt.title("Training Loss of DNN", fontsize=14)
# plt.xlabel("Time Frames", fontsize=12)
# plt.ylabel("Loss", fontsize=12)
# plt.grid(True, alpha=0.5)
# plt.legend()
# plt.savefig('./Training_Eng.pdf')
# -------------------------------------------------------------------------------------------

# # 读取数据
# data_file_TI1 = './user:20_uav:3_different_TI_exp/EngCostDuringTraining_TI:1_MemS:384.csv'
# data_file_TI5 = './user:20_uav:3_different_TI_exp/EngCostDuringTraining_TI:5_MemS:384.csv'
# data_file_TI10 = './user:20_uav:3_different_TI_exp/EngCostDuringTraining_TI:10_MemS:384.csv'
# data_file_TI20 = './user:20_uav:3_different_TI_exp/EngCostDuringTraining_TI:20_MemS:384.csv'
# data_file_TI50 = './user:20_uav:3_different_TI_exp/EngCostDuringTraining_TI:50_MemS:384.csv'
# data_file_TI100 = './user:20_uav:3_different_TI_exp/EngCostDuringTraining_TI:100_MemS:384.csv'

# df = 2000
# t1 = read_csv(data_file_TI1, diff=df)
# t5 = read_csv(data_file_TI5, diff=df)
# t10 = read_csv(data_file_TI10, diff=df)
# t20 = read_csv(data_file_TI20, diff=df)
# t50 = read_csv(data_file_TI50, diff=df)
# t100 = read_csv(data_file_TI100, diff=df)

# # 开始画图
# plt.figure(figsize=(10, 5))
# x = np.arange(1, len(t1) + 1, step=1)
# plt.plot(x, t1, label='training interval = 1')
# plt.plot(x, t5, label='training interval = 5')
# plt.plot(x, t10, label='training interval = 10')
# plt.plot(x, t20, label='training interval = 20')
# plt.plot(x, t50, label='training interval = 50')
# plt.plot(x, t100, label='training interval = 100')
# plt.axhline(y=1384.5791015625, color='grey', linestyle='dashdot', label='REF')

# plt.title("Impact of Training Interval on Energy Cost", fontsize=14)
# plt.xlabel("Time Frames", fontsize=12)
# plt.ylabel("Energy cost", fontsize=12)
# plt.grid(True, alpha=0.5)
# plt.legend()
# # 显示图形
# plt.savefig('./eng_cost_diff_TI.pdf')
# -------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------
# mem0_5 = './user:20_uav:3_MemSize_Exp/EngCostDuringTraining_TI:10_MemS:128.csv'
# mem1 = './user:20_uav:3_MemSize_Exp/EngCostDuringTraining_TI:10_MemS:256.csv'
# mem1_5 = './user:20_uav:3_MemSize_Exp/EngCostDuringTraining_TI:10_MemS:384.csv'
# mem2 = './user:20_uav:3_MemSize_Exp/EngCostDuringTraining_TI:10_MemS:512.csv'
# mem4 = './user:20_uav:3_MemSize_Exp/EngCostDuringTraining_TI:10_MemS:1024.csv'
# mem8 = './user:20_uav:3_MemSize_Exp/EngCostDuringTraining_TI:10_MemS:2048.csv'

# m0_5 = read_csv(mem0_5)
# m1 = read_csv(mem1)
# m1_5 = read_csv(mem1_5)
# m2 = read_csv(mem2)
# m4 = read_csv(mem4)
# m8 = read_csv(mem8)

# plt.figure(figsize=(10, 5))
# x = np.arange(1, len(m1) + 1, step=1)
# plt.plot(x, m0_5, label='Memory size = 0.5 x batch size')
# plt.plot(x, m1, label='Memory size = 1 x batch size')
# plt.plot(x, m1_5, label='Memory size = 1.5 x batch size')
# plt.plot(x, m2, label='Memory size = 2 x batch size')
# plt.plot(x, m4, label='Memory size = 4 x batch size')
# plt.plot(x, m8, label='Memory size = 8 x batch size')
# plt.axhline(y=1384.5791015625, color='grey', linestyle='dashdot', label='REF')

# plt.title("Impact of Memory Size on Energy Cost", fontsize=14)
# plt.xlabel("Time Frames", fontsize=12)
# plt.ylabel("Energy cost", fontsize=12)
# plt.grid(True, alpha=0.5)
# plt.legend()
# plt.savefig('./eng_cost_diff_mem.pdf')
# -------------------------------------------------------------------------------------------
