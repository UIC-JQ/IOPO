import csv 
import numpy as np # import numpy

# Implementated based on the PyTorch 
from memoryPyTorch import MemoryDNN
# from opt3 import whale
import torch

from tqdm import tqdm

from dataclass import DataConfig
from opt3 import whale

def save_to_txt(rate_his, file_path):
    with open(file_path, 'w') as f:
        for rate in rate_his:
            f.write("%s \n" % rate)

def load_from_csv(file_path=None, data_type=None):
    with open(file_path, mode='r') as f:
        reader = csv.reader(f, delimiter=',')
        data = []

        for row in reader:
            data.append(row)
            
        data = np.array(data, dtype=data_type)
    
    return data

def eng_cost_wrapper_func(records, data_config):
    def inner(idx, allocate_plan):
        return whale(records[idx], allocate_plan, data_config=data_config, need_stats=False)
    
    return inner

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
    number_of_iter     = 1000                                                   # number of time frames
    input_feature_size = None                                                   # dim of training sample
    output_y_size      = number_of_user * (number_of_uav + 1)                   # 神经网络输出dim
    # number_of_uav + 1是因为 [0, 1, 2, 3], 0表示本地，1-3为无人机编号
    cvt_output_size    = number_of_user * number_of_uav
    # 用于生成答案数组 (由0，1)构成

    # ----------------------------
    # Memory = 1024                  # capacity of memory structure
    # decoder_mode = 'MP'          # the quantization mode could be 'OP' (Order-preserving) or 'KNN'
    # K = N                        # initialize K = N
    # Delta = 32                   # Update interval for adaptive K


    # Load training data
    X_feature_file       = 'TRAINING_NumOfUser:{}_NumOfUAV:{}_feature.csv'.format(number_of_user, number_of_uav)
    Y_ans_file           = 'TRAINING_NumOfUser:{}_NumOfUAV:{}_solution.csv'.format(number_of_user, number_of_uav)
    model_save_path      = 'MODEL_NumOfUser:{}_NumOfUAV:{}.pt'.format(number_of_user, number_of_uav)
    Y_eng_cost_save_path = 'TRAINING_NumOfUser:{}_NumOfUAV:{}_energy_cost.csv'.format(number_of_user, number_of_uav)
    ENV_file_path        = 'TRAINING_NumOfUser:{}_NumOfUAV:{}_record.csv'.format(number_of_user, number_of_uav)

    X = load_from_csv(file_path=X_feature_file, data_type=float)
    input_feature_size = X[0].size

    Y = load_from_csv(file_path=Y_ans_file, data_type=int)

    ENG_COST = load_from_csv(file_path=Y_eng_cost_save_path, data_type=float)

    ENV      = load_from_csv(file_path=ENV_file_path, data_type=float)

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
                    dropout=0.2,
                    data = data_pairs,
                    split_len = number_of_uav + 1,
                    convert_output_size=cvt_output_size,
                    data_config=data_config,
                    data_eng_cost=ENG_COST
                    )

    energy_cost_function = eng_cost_wrapper_func(ENV, data_config)
    # start training
    for i in tqdm(range(number_of_iter)):
        if i % 100 == 0 and i != 0:
            model.train(flag_regenerate_better_sol=True, eng_cost_func=energy_cost_function)
        else:
            model.train()

        # model.train()
        

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