# Implementated based on the PyTorch 
from Model_LSTM import LSTM_Model
from Model_LSTM_IMP import Model_LSTM_IMP
from Model_MLP import MLP

import torch

from tqdm import tqdm

from dataclass import DataConfig
from opt3 import whale
from util import load_from_csv, setup_seed

import argparse

def eng_cost_wrapper_func(records, data_config):
    """
    计算给定allocation plan的energy cost.
    """
    def inner(idx, allocate_plan):
        return whale(records[idx], allocate_plan, data_config=data_config, need_stats=False)
    
    return inner

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--nnModel', type=str, help='使用哪一个NN模型, choose from {MLP, LSTM, LSTM_ATT}')
    parser.add_argument('--uavNumber', type=int, help='uav的数量')
    parser.add_argument('--userNumber', type=int, help='user的数量')
    parser.add_argument('--batch_size', type=int, help='batch size的大小')
    parser.add_argument('--hidden_dim', type=int, help='hidden dimension的大小')
    parser.add_argument('--num_of_iter', type=int, help='训练的轮数')
    parser.add_argument('--drop_out', type=float, help='drop out概率')
    args = parser.parse_args()

    # 创建数据配置
    number_of_uav = args.uavNumber                          # numbers of UAVs 
    number_of_user = args.userNumber                        # number of users
    inner_path = 'NumOfUser:{}_NumOfUAV:{}'.format(number_of_user, number_of_uav)
    data_config = DataConfig(load_config_from_path='CONFIG_' + inner_path + '.json')

    # 训练NN配置
    number_of_iter     = args.num_of_iter                                       # number of time frames
    batch_size         = args.batch_size 
    train_per_step     = 10                                                     # 每添加相应个数据后，训练网络一次
    input_feature_size = None                                                   # dim of training sample
    output_y_size      = number_of_user * (number_of_uav + 1)                   # 神经网络输出dim, number_of_uav + 1 是因为 [0, 1, 2, 3], 0表示本地，1-3为无人机编号, +1是为了多出来的0
    cvt_output_size    = number_of_user * number_of_uav                         # 用于生成答案数组 (由0，1)构成
    Memory             = batch_size * 4                                         # 设置Memory size大小

    # ---------------------------------------------------------------------------------
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
    # ---------------------------------------------------------------------------------

    # create model
    model_name = args.nnModel.upper()
    if model_name == 'MLP':
        model = MLP
    elif model_name == 'LSTM':
        model = LSTM_Model
    else:
        model = Model_LSTM_IMP

    model = model(input_feature_size,
                  output_size=output_y_size,
                  hidden_feature_size=args.hidden_dim,
                  learning_rate = 0.001,
                  training_interval=train_per_step,
                  batch_size=batch_size,
                  dropout=args.drop_out,
                  split_len = number_of_uav + 1,
                  convert_output_size=cvt_output_size,
                  data_config=data_config,
                  memory_size=Memory,
    )

    # 定义energy cost计算函数
    energy_cost_function = eng_cost_wrapper_func(Record, data_config)

    # -------------------------------------------------------
    # start training
    for i in tqdm(range(number_of_iter)):
        idx = i % Num_of_training_pairs
        input_feature = X[idx]

        # eng_cost, new_y = model.decode(input_feature, K=20, eng_compute_func=energy_cost_function, idx=idx)
        # # 如果存在能耗更低的解
        # if eng_cost < ENG_COST[idx]:
        #     print('Regenerate a better solution, cost', eng_cost)
        #     print(new_y)
        #     print('original solution, cost', ENG_COST[idx])
        #     print(Y[idx])
        #     ENG_COST[idx, :] = eng_cost
        #     Y[idx, :]        = torch.Tensor(new_y)

        model.encode(feature=input_feature, y=Y[idx])
       
        
    model.plot_cost()
    # save model parameters:
    model_save_path = 'MODEL_NumOfUser:{}_NumOfUAV:{}.pt'.format(number_of_user, number_of_uav)
    model.save_model(model_save_path)