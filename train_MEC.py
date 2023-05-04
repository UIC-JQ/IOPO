import csv 
import numpy as np # import numpy

# Implementated based on the PyTorch 
from memoryPyTorch import MemoryDNN
# from opt3 import whale
import torch

from tqdm import tqdm

from dataclass import DataConfig

def save_to_txt(rate_his, file_path):
    with open(file_path, 'w') as f:
        for rate in rate_his:
            f.write("%s \n" % rate)

if __name__ == "__main__":
    '''
        This algorithm generates K modes from DNN, and chooses with largest
        reward. The mode with largest reward is stored in the memory, which is
        further used to train the DNN.
        Adaptive K is implemented. K = max(K, K_his[-memory_size])
    '''
    # 创建数据配置
    number_of_uav = 6                         # numbers of UAVs 
    number_of_user = 3                        # number of users
    inner_path = 'NumOfUser:{}_NumOfUAV:{}'.format(number_of_user, number_of_uav)
    data_config = DataConfig(load_config_from_path='CONFIG_' + inner_path + '.json')

    # 训练NN配置
    number_of_iter     = 10000                    # number of time frames
    input_feature_size = None                 # dim of training sample
    output_y_size      = number_of_user * number_of_uav                    # 神经网络输出dim

    # ----------------------------
    # Memory = 1024                  # capacity of memory structure
    # decoder_mode = 'MP'          # the quantization mode could be 'OP' (Order-preserving) or 'KNN'
    # K = N                        # initialize K = N
    # Delta = 32                   # Update interval for adaptive K


    # Load training data
    X_feature_file  = 'TRAINING_NumOfUser:3_NumOfUAV:6_feature.csv'
    Y_ans_file      = 'TRAINING_NumOfUser:3_NumOfUAV:6_solution.csv'
    model_save_path = 'MODEL_NumOfUser:{}_NumOfUAV:{}.pt'.format(number_of_user, number_of_uav)

    with open(X_feature_file, mode='r') as file:
        reader = csv.reader(file, delimiter=',')
        X = []

        for row in reader:
            X.append(row)
            
        X = np.array(X, dtype=float)
        input_feature_size = X[0].size

    with open(Y_ans_file, mode='r') as file:
        reader = csv.reader(file, delimiter=',')
        Y = []

        for row in reader:
            Y.append(row)
            
        Y = np.array(Y, dtype=int)

    # 构造(X, Y) data pairs，用于训练
    data_pairs = np.concatenate((X, Y), axis=1)
    
    # create model
    model = MemoryDNN(
                    input_feature_size,
                    output_size=output_y_size,
                    hidden_feature_size=256,
                    learning_rate = 0.001,
                    training_interval=1,
                    batch_size=256,
                    dropout=0.1,
                    data = data_pairs
                    )


    # min_energy = []
    #rate_his_ratio = []
    # mode_his = [] #1,0分配结果
    # k_idx_his = []
    # K_his = []

    # start training
    for i in tqdm(range(number_of_iter)):
        # if i > 0 and i % Delta == 0: #当i为delta的倍数时
        #     # index counts from 0
        #     if Delta > 1:#取一些数中的最大值
        #         max_k = max(k_idx_his[-Delta:-1])+1
        #     else:
        #         max_k = k_idx_his[-1]+1
        #     K = max(max_k + 1, N)

        # 读取第I行训练数据
        # i_idx = i % len(X)
        # x_i = X[i_idx,:]
        # y_i = Y[i_idx,:]


        # model.encode(x_i, y_i)
        model.train()
        
        # ------------------------------------------
        # TODO: remove below code:
        # the action selection must be either 'OP' or 'KNN' or 'MP'
        # m_list = mem.decode(x_training_feature, K, decoder_mode)  #m_list为DNN算出的1，0结果
        # m_list = mem.decode(h, out, decoder_mode)  #m_list为DNN算出的1，0结果
        # mem.encode(h, min_energy_cost_selection)


        # r_list = [] #memory size 储存最小能耗
        # count = 0
        # count2 = 0
        # for m in m_list:
        #     r_list.append(whale(h, m, data_config)[1]) #循环一条条m带入计算
        # # encode the mode with 最小能耗

        # eng, b0 = random_sample_lower_eng_cost_plan(h)
        # # print('fitness of best solution (after random selection process) = {}'.format(eng))

        # if np.argmin(r_list) < eng:
        #     mem.encode(h, m_list[np.argmin(r_list)])  #加入优化结果重新训练
        # else:
        #     mem.encode(h, b0) 
        

        # # the main code for DROO training ends here

        # # the following codes store some interested metrics for illustrations
        # # memorize the 最小能耗
        # min_energy.append(np.min(r_list))
        # # rate_his_ratio.append(rate_his[-1] / rate[i_idx][0])
        # # record the index of largest reward
        # k_idx_his.append(np.argmin(r_list))
        # # record K in case of adaptive K
        # K_his.append(K)

        # min_idx = np.argmin(r_list)
        # mode_his.append(m_list[np.argmin(r_list)])
        

    model.plot_cost()
    model.save_model(model_save_path)
    # save data into txt
    # path1 = "/Users/Helen/Documents/Mphi/code/UAV-IRS V2/result/"
    
    ## save_to_txt(rate_his_ratio, path1+"rate_his_ratio.txt")

    # save_path = path1 + 'model_param.pt'
    # save_to_txt(k_idx_his, path1+"k_idx_his.txt")
    # save_to_txt(K_his, path1+"K_his.txt")
    # save_to_txt(mem.cost_his, path1+"cost_his.txt")
    # save_to_txt(r_list, path1+"energy.txt")
    # save_to_txt(mode_his, path1+"mode_his.txt")