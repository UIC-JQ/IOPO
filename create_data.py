import numpy as np
import random
import csv
from tqdm import tqdm

def save_to_csv(data, file_path):
    with open(file_path, mode='w+', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)

def random_location_generate(n, user, mode):
    #坐标范围:
    lx = 600
    ly = 800

    la = [] #训练数据
    print("the dim of one sample is %d"%(user * 6))

    # features:
    for _ in tqdm(range(n)):
        lm = []

        # 生成用户坐标：(x, y, z)
        for _ in range(user):
            lm.append(np.random.uniform(0, lx))
            lm.append(np.random.uniform(0, ly))
            lm.append(0)

        # 生成task info   
        for _ in range(user):
            data = round(np.random.uniform(3e2, 5e2)) #用户平均任务大小 Mbit/slot单位时间多少Mbit
            lm.append(data)

            cpb = np.random.randint(40,150)*1e-3 # 单位比特计算次数为 40-150 cycles
            data = np.round(data*cpb)#任务需要的cpu数量
            lm.append(data)

            lm.append(np.random.randint(np.ceil(data*1e6/125000),np.ceil(data*1e6/125000)+1000))#任务可以接受的延迟 slot

        la.append(lm)

    if mode == 'train':
        save_to_csv(la, "./training_data/trainData_userNumber=%d_n=%d.csv" % (user, n))
    elif mode == 'test':
        save_to_csv(la, "./test_data/testData_userNumber=%d_n=%d.csv" % (user, n))

    # print(la)
    # la = np.array(la)
    # print(la.shape)
        
    return

if __name__ == '__main__':
    # random_location_generate(n=5000, user=3, mode='test')
    random_location_generate(n=10000, user=3, mode='train')