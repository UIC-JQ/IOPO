import csv 
import numpy as np # import numpy

# Implementated based on the PyTorch 
from memoryPyTorch import MemoryDNN
from opt3 import whale
import torch

import time
from tqdm import tqdm

from dataclass import DataConfig

def save_to_txt(rate_his, file_path):
    with open(file_path, 'w') as f:
        for rate in rate_his:
            f.write("%s \n" % rate)

def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)

def constrain(m,uav):
    m = np.array(m)
    m = m.reshape(-1, uav)
    sums = np.sum(m,axis=1)
    result = np.all(sums <= 1)
    return result

def random_sample_lower_eng_cost_plan(h):
    """
    date: 2023/4/30
    author: Yu, Jianqiu
    """
    # print('using random')
    config_sol_size = U * M
    config_random_times = 5
        
    # all 0 vector:
    sol_ = np.zeros(config_sol_size, dtype=int)
    # compute [case: all local computation] energy:
    _, e_best = whale(h, sol_, data_config)

    for _ in range(config_random_times):
        random_sol = np.zeros(config_sol_size, dtype=int)

        
        for i in range(0, config_sol_size, M):
            use_one_flag = np.random.randint(0, 2)

            if use_one_flag == 1:
                idx = np.random.randint(i, i + M)
                random_sol[idx] = 1
        
        # print('random solution:', random_sol)

        _, new_energy = whale(h, random_sol, data_config)#重新算结果

        if new_energy < e_best:
            e_best = new_energy
            sol_ = random_sol

    return e_best, sol_

if __name__ == "__main__":
    '''
        This algorithm generates K modes from DNN, and chooses with largest
        reward. The mode with largest reward is stored in the memory, which is
        further used to train the DNN.
        Adaptive K is implemented. K = max(K, K_his[-memory_size])
    '''
    M = 6                        # numbers of UAVs 
    U = 3                        # number of users
    N = U*M                      # dim of traning sample
    n = 100                      # number of time frames
    K = N                        # initialize K = N
    out = M*U                    # 神经网络输出dim
    decoder_mode = 'MP'          # the quantization mode could be 'OP' (Order-preserving) or 'KNN'
    Memory = 1024                # capacity of memory structure
    Delta = 32                   # Update interval for adaptive K

    data_config = DataConfig(n_of_user=U, n_of_uav=M)

    print('#input dim = %d, #sample=%d,decoder = %s, Memory = %d, Delta = %d'%(N,n,decoder_mode, Memory, Delta))
    # Load training data
    path = './training_data/data_%d.csv' % U
    with open(path, mode='r') as file:
        # 创建CSV读取器，指定分隔符为逗号
        reader = csv.reader(file, delimiter=',')
        # 读取CSV文件的数据到一个列表中
        data = []

        for row in reader:
            data.append(row)
            
        # load all
        # for row in reader:
            # X = data.append(row)
        # for row in reader_answer:
            # Y = data.append(row)

        location = np.array(data, dtype=float)

    # create model
    mem = MemoryDNN(net = [K, 120, 80, out],
                    learning_rate = 0.001,
                    training_interval=10,
                    batch_size=128,
                    memory_size=Memory,
                    dropout=0.1
                    )


    min_energy = []
    #rate_his_ratio = []
    mode_his = [] #1,0分配结果
    k_idx_his = []
    K_his = []

    # start training
    # TODO: train total of n epochs
    for i in tqdm(range(n)):
        if i > 0 and i % Delta == 0: #当i为delta的倍数时
            # index counts from 0
            if Delta > 1:#取一些数中的最大值
                max_k = max(k_idx_his[-Delta:-1])+1
            else:
                max_k = k_idx_his[-1]+1
            K = max(max_k + 1, N)

        # 读取第I行训练数据
        i_idx = i % len(location)
        h = location[i_idx,:]
        
        # the action selection must be either 'OP' or 'KNN' or 'MP'
        m_list = mem.decode(h, K, decoder_mode)  #m_list为DNN算出的1，0结果
        # TODO:
        # m_list = mem.decode(h, out, decoder_mode)  #m_list为DNN算出的1，0结果
        # mem.encode(h, min_energy_cost_selection)


        r_list = [] #memory size 储存最小能耗
        count = 0
        count2 = 0
        for m in m_list:
            r_list.append(whale(h, m, data_config)[1]) #循环一条条m带入计算
        # encode the mode with 最小能耗

        eng, b0 = random_sample_lower_eng_cost_plan(h)
        # print('fitness of best solution (after random selection process) = {}'.format(eng))

        if np.argmin(r_list) < eng:
            mem.encode(h, m_list[np.argmin(r_list)])  #加入优化结果重新训练
        else:
            mem.encode(h, b0) 
        

        # the main code for DROO training ends here

        # the following codes store some interested metrics for illustrations
        # memorize the 最小能耗
        min_energy.append(np.min(r_list))
        # rate_his_ratio.append(rate_his[-1] / rate[i_idx][0])
        # record the index of largest reward
        k_idx_his.append(np.argmin(r_list))
        # record K in case of adaptive K
        K_his.append(K)

        min_idx = np.argmin(r_list)
        mode_his.append(m_list[np.argmin(r_list)])
        

    mem.plot_cost()
    # save data into txt
    path1 = "/Users/Helen/Documents/Mphi/code/UAV-IRS V2/result/"
    
    ## save_to_txt(rate_his_ratio, path1+"rate_his_ratio.txt")

    save_path = path1 + 'model_param.pt'
    save_model(mem, save_path)
    save_to_txt(k_idx_his, path1+"k_idx_his.txt")
    save_to_txt(K_his, path1+"K_his.txt")
    save_to_txt(mem.cost_his, path1+"cost_his.txt")
    save_to_txt(r_list, path1+"energy.txt")
    save_to_txt(mode_his, path1+"mode_his.txt")
        
        


# ----------------------------------------------------
# def check_if_correct(generated):
    # for i in range(0, len(generated), M):
        # assert sum(generated[i: i + M]) <= 1, 'error: {}'.format(generated)
    
# for row in m_list:
    # check_if_correct(row)