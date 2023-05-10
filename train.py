# Implementated based on the PyTorch 
from Model import MemoryDNN, LSTM_Model

import torch

from tqdm import tqdm

from dataclass import DataConfig
from opt3 import whale
from util import load_from_csv, setup_seed

def save_to_txt(rate_his, file_path):
    with open(file_path, 'w') as f:
        for rate in rate_his:
            f.write("%s \n" % rate)

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
    number_of_uav = 6                          # numbers of UAVs 
    number_of_user = 10                         # number of users
    inner_path = 'NumOfUser:{}_NumOfUAV:{}'.format(number_of_user, number_of_uav)
    data_config = DataConfig(load_config_from_path='CONFIG_' + inner_path + '.json')

    # 训练NN配置
    number_of_iter     = 40000                                                  # number of time frames
    train_per_step     = 10                                                      # 每添加相应个数据后，训练网络一次
    input_feature_size = None                                                   # dim of training sample
    output_y_size      = number_of_user * (number_of_uav + 1)                   # 神经网络输出dim
    # number_of_uav + 1 是因为 [0, 1, 2, 3], 0表示本地，1-3为无人机编号, +1是为了多出来的0
    cvt_output_size    = number_of_user * number_of_uav                         # 用于生成答案数组 (由0，1)构成
    batch_size         = 256 

    # ----------------------------
    # TODO: re-implement
    Memory             = batch_size * 4               # capacity of memory structure
    # decoder_mode = 'MP'                             # the quantization mode could be 'OP' (Order-preserving) or 'KNN'
    # K = N                                           # initialize K = N
    # Delta = 32                                      # Update interval for adaptive K

    # Load training data
    X_feature_file        = 'TRAINING_NumOfUser:{}_NumOfUAV:{}_feature.csv'.format(number_of_user, number_of_uav)
    Y_ans_file            = 'TRAINING_NumOfUser:{}_NumOfUAV:{}_solution.csv'.format(number_of_user, number_of_uav)
    Y_eng_cost_save_path  = 'TRAINING_NumOfUser:{}_NumOfUAV:{}_energy_cost.csv'.format(number_of_user, number_of_uav)
    ENV_file_path         = 'TRAINING_NumOfUser:{}_NumOfUAV:{}_record.csv'.format(number_of_user, number_of_uav)

    X                     = torch.Tensor(load_from_csv(file_path=X_feature_file, data_type=float))         # 读取input feature
    input_feature_size    = X.shape[-1]                                                                    # 获取input feature的维度
    Num_of_training_pairs = len(X)                                                                         # 获取训练数据量

    Y                     = torch.Tensor(load_from_csv(file_path=Y_ans_file, data_type=int))               # 读取reference answer
    ENG_COST              = load_from_csv(file_path=Y_eng_cost_save_path, data_type=float)                 # 读取reference answer的energy cost
    Record                = load_from_csv(file_path=ENV_file_path, data_type=float)                        # 读取环境状态

    # create model
    model_name = LSTM_Model
    # model_name = MemoryDNN

    model = model_name(input_feature_size,
                       output_size=output_y_size,
                       hidden_feature_size=512,
                       learning_rate = 0.001,
                       training_interval=train_per_step,
                       batch_size=batch_size,
                       dropout=0.2,
                       split_len = number_of_uav + 1,
                       convert_output_size=cvt_output_size,
                       data_config=data_config,
                       memory_size=Memory,
    )

    # 定义energy cost计算函数
    energy_cost_function = eng_cost_wrapper_func(Record, data_config)
    # 设置随机种子
    # setup_seed()

    # start training
    # -------------------------------------------------------
    for i in tqdm(range(number_of_iter)):
        idx = i % Num_of_training_pairs
        input_feature = X[idx]

        # eng_cost, new_y = model.decode(input_feature, K=5, eng_compute_func=energy_cost_function, idx=idx)
        # # 如果存在能耗更低的解
        # if eng_cost < ENG_COST[idx]:
        #     ENG_COST[idx, :] = eng_cost
        #     Y[idx, :]        = torch.Tensor(new_y)

        model.encode(feature=input_feature, y=Y[idx])
       
        
    model.plot_cost()
    # save model parameters:
    model_save_path = 'MODEL_NumOfUser:{}_NumOfUAV:{}.pt'.format(number_of_user, number_of_uav)
    model.save_model(model_save_path)