import numpy as np
from dataclass import DataConfig
from opt3 import whale
from tqdm import tqdm
import argparse

from Model_LSTM import LSTM_Model
from Model_MLP import MLP
from Model_LSTM_IMP import Model_LSTM_IMP
from util import load_from_csv, setup_seed


def try_method(X, method, method_name, **kws):
    """
    测试模型method, 并统计相关结果
    """
    avg_eng_cost, overtime_records_ratio, avg_overtime_ratio, avg_overtime_people = method(X, data_config, **kws)
    print('--> 方法名: [%s]' % method_name)
    print('平均每条record的energy cost:', avg_eng_cost)
    print('所有record中，存在超时user的record比例: {:.3f}%'.format(overtime_records_ratio * 100))
    print('所有存在超时user的record中，平均超时人数占总人数的比例: {:.3f}%'.format(avg_overtime_ratio * 100))
    print('所有存在超时user的record中，平均超时人数为', avg_overtime_people)

def print_plan(allocate_plan, data_config):
    """
    打印allocation plan
    """
    ans = []
    base = 0
    allocate_plan = np.array(allocate_plan)

    # 将format为[0，0，1]的allocation_plan -> 转换为更可读的allocation plan [3]
    for _ in range(data_config.user_number):
        sol = allocate_plan[base: base + data_config.uav_number]
        idx = np.where(sol == 1)[0]
        if not idx:
            ans.append(0)
        else:
            ans.append(idx[0] + 1)
        base += data_config.uav_number

    # 打印结果 
    print(ans)

def allocate_plan_all_local(record, data_config: DataConfig, **kws):
    return np.zeros(data_config.user_number * data_config.uav_number)

def allocate_plan_all_upload_random(record, data_config: DataConfig, **kws):
    _, __allocate_plan = data_config.random_sample_lower_eng_cost_plan(record,
                                                                       K=kws['K'],
                                                                       exclude_local_options=True)
    if kws['print_plan']:
        print_plan(__allocate_plan, data_config)

    return __allocate_plan

def allocate_plan_local_and_upload_random(record, data_config: DataConfig, **kws):
    _, __allocate_plan = data_config.random_sample_lower_eng_cost_plan(record,
                                                                       K=kws['K'],
                                                                       exclude_local_options=False)
    if kws['print_plan']:
        print_plan(__allocate_plan, data_config)

    return __allocate_plan

def allocate_plan_NN_model(record, data_config: DataConfig, **kws):
    model = kws['model']
    data_idx = kws['data_idx']
    X = kws['input_feature'][data_idx,:]
    _, __allocate_plan = model.generate_answer(X, data_config)

    if kws['print_plan']:
        print_plan(__allocate_plan, data_config)

    return __allocate_plan

def allocate_plan_by_local_compute_t_and_uav_comp_speed(record, data_config: DataConfig, **kws):
    data_idx = kws['data_idx']
    local_time_ranking = [eval(s) for s in kws['ranking'][data_idx]]
    _, _, __allocate_plan = data_config.allocate_by_local_time_and_uav_comp_speed(record, local_time_ranking)

    if kws['print_plan']:
        print_plan(__allocate_plan, data_config)

    return __allocate_plan

def compare_method(allocate_plan_generation_method):
    def inner(X, data_config, **kws):
        total_eng_cost = 0                      # 总的energy cost
        total_overtime_records_num = 0          # 有多少测试数据的分配方案中，会存在有超时的用户
        store_overtime_ratio = []               # 存在超时的分配方案中，超时用户的比例。
        store_overtime_people = []              # 存在超时的分配方案中，超时用户的人数。

        for idx, record in enumerate(tqdm(X)):
            # 用给定方法生成一个 allocate plan
            __allocate_plan = allocate_plan_generation_method(record, data_config, data_idx=idx, **kws)
            # 计算energy cost
            _, energy, overtime_logs = whale(record, __allocate_plan, data_config, need_stats=True, optimize_phi=True)

            if overtime_logs:
            # 统计超时数据stat
            # 如果这条record有超时
                total_overtime_records_num += 1
                store_overtime_ratio.append(len(overtime_logs) / data_config.user_number)
                store_overtime_people.append(len(overtime_logs))

            total_eng_cost += energy

        # 计算stats: 
        avg_eng_cost                    = total_eng_cost / len(X)
        overtime_records_ratio          = total_overtime_records_num / len(X)
        avg_overtime_ratio              = sum(store_overtime_ratio) / len(store_overtime_ratio) if len(store_overtime_ratio) else 0
        avg_overtime_people             = sum(store_overtime_people) / len(store_overtime_people) if len(store_overtime_people) else 0

        return avg_eng_cost, overtime_records_ratio, avg_overtime_ratio, avg_overtime_people
    
    return inner


if __name__ == '__main__':
    # SETTINGS:
    parser = argparse.ArgumentParser()
    parser.add_argument('--nnModel', type=str, help='使用哪一个NN模型, choose from {MLP, LSTM, LSTM_ATT}')
    parser.add_argument('--uavNumber', type=int, help='uav的数量')
    parser.add_argument('--userNumber', type=int, help='user的数量')
    parser.add_argument('--test_NN_only', action='store_true', help='user的数量', default=False)
    args = parser.parse_args()

    # 创建数据配置
    number_of_uav = args.uavNumber                          # numbers of UAVs 
    number_of_user = args.userNumber                        # number of users
    inner_path = 'NumOfUser:{}_NumOfUAV:{}'.format(number_of_user, number_of_uav)
    data_config = DataConfig(load_config_from_path='./Config/CONFIG_' + inner_path + '.json')

    # ---------------------------------------------------------------------------------------------------------
    # LOAD Test data:
    path = './Dataset/TESTING_NumOfUser:{}_NumOfUAV:{}_record.csv'.format(number_of_user, number_of_uav)
    Record = load_from_csv(path, data_type=float)
    
    X_feature_file = './Dataset/TESTING_NumOfUser:{}_NumOfUAV:{}_feature.csv'.format(number_of_user, number_of_uav)
    feature = load_from_csv(X_feature_file, data_type=float)

    local_rank_file = './Dataset/TESTING_NumOfUser:{}_NumOfUAV:{}_local_comp_time_ranking.csv'.format(number_of_user, number_of_uav)
    local_time_rankings = load_from_csv(local_rank_file)
    # ---------------------------------------------------------------------------------------------------------
    setup_seed()                     # 设置固定的随机种子

    # 选择模型
    model_name = args.nnModel.upper()
    if model_name == 'MLP':
        print('Loading model MLP')
        model = MLP
    elif model_name == 'LSTM':
        print('Loading model Naive LSTM')
        model = LSTM_Model
    else:
        print('Loading model LSTM w/ Attention')
        model = Model_LSTM_IMP

    model = model.load_model('./Saved_model/MODEL_{}_NumOfUser:{}_NumOfUAV:{}.pt'.format(model_name, number_of_user, number_of_uav))
    try_method(Record, 
               method=compare_method(allocate_plan_NN_model),
               method_name='NN Model : {}'.format(str(model_name)),
               model=model,
               input_feature=feature,
               print_plan=False)
    
    if args.test_NN_only:
        print('[Config] Only test NN method only.')
        exit(0)

    print('-' * 50)
    try_method(Record,
               method=compare_method(allocate_plan_by_local_compute_t_and_uav_comp_speed),
               method_name='Greedy (by Local Compute time and uav Compute speed)',
               ranking=local_time_rankings,
               print_plan=False)
    print('-' * 50)
    try_method(Record, compare_method(allocate_plan_all_upload_random), 'ALL UPLOAD OPTIMIZED RANDOM (K=1)', K=1, print_plan=False)
    try_method(Record, compare_method(allocate_plan_all_upload_random), 'ALL UPLOAD OPTIMIZED RANDOM (K=10)', K=10, print_plan=False)
    try_method(Record, compare_method(allocate_plan_all_upload_random), 'ALL UPLOAD OPTIMIZED RANDOM (K=50)', K=50, print_plan=False)
    print('-' * 50)
    try_method(Record, compare_method(allocate_plan_local_and_upload_random), '(LOCAL + UPLOAD) OPTIMIZED RANDOM (K=1)', K=1, print_plan=False)
    try_method(Record, compare_method(allocate_plan_local_and_upload_random), '(LOCAL + UPLOAD) OPTIMIZED RANDOM (K=10)', K=10, print_plan=False)
    try_method(Record, compare_method(allocate_plan_local_and_upload_random), '(LOCAL + UPLOAD) OPTIMIZED RANDOM (K=50)', K=50, print_plan=False)
    print('-' * 50)
    try_method(Record, compare_method(allocate_plan_all_local), 'ALL LOCAL', K=1, print_plan=False)