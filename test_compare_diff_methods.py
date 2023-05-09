import csv
import numpy as np
from dataclass import DataConfig
from opt3 import whale
from tqdm import tqdm

from memoryPyTorch import MemoryDNN
from util import load_from_csv, setup_seed

def try_method(X, method, method_name, **kws):
    avg_eng_cost, overtime_records_ratio, avg_overtime_ratio, avg_overtime_people = method(X, data_config, **kws)
    print('--> 方法名: [%s]' % method_name)
    print('平均每条record的energy cost:', avg_eng_cost)
    print('所有record中，存在超时user的record比例: {:.3f}%'.format(overtime_records_ratio * 100))
    print('所有存在超时user的record中，平均超时人数占总人数的比例: {:.3f}%'.format(avg_overtime_ratio * 100))
    print('所有存在超时user的record中，平均超时人数为', avg_overtime_people)

def allocate_plan_all_local(record, data_config, **kws):
    return np.zeros(data_config.user_number * data_config.uav_number)

def allocate_plan_all_upload_random(record, data_config, **kws):
    _, __allocate_plan = data_config.random_sample_lower_eng_cost_plan(record,
                                                                       K=kws['K'],
                                                                       exclude_local_options=True)
    return __allocate_plan

def allocate_plan_local_and_upload_random(record, data_config, **kws):
    _, __allocate_plan = data_config.random_sample_lower_eng_cost_plan(record,
                                                                       K=kws['K'],
                                                                       exclude_local_options=False)
    return __allocate_plan

def allocate_plan_NN_model(record, data_config, **kws):
    model = kws['model']
    data_idx = kws['data_idx']
    X = kws['input_feature'][data_idx,:]
    Y = model.generate_answer(X, data_config)

    return Y


def compare_method(f):
    def inner(X, data_config, **kws):
        total_eng_cost = 0
        total_overtime_records_num = 0
        store_overtime_ratio = []
        store_overtime_people = []

        for idx, record in enumerate(tqdm(X)):
            # 用给定方法生成一个 allocate plan
            __allocate_plan = f(record, data_config, data_idx=idx, **kws)
            # 计算energy cost
            _, energy, overtime_logs = whale(record, __allocate_plan, data_config, need_stats=True)
            # print('Allocation Plan: {}, Eng cost: {}'.format(__allocate_plan, energy))

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
    number_of_uav = 6                         # numbers of UAVs 
    number_of_user = 3                        # number of users
    inner_path = 'NumOfUser:{}_NumOfUAV:{}'.format(number_of_user, number_of_uav)
    data_config = DataConfig(load_config_from_path='CONFIG_' + inner_path + '.json')

    # LOAD Test data:
    path = 'TESTING_NumOfUser:{}_NumOfUAV:{}_record.csv'.format(number_of_user, number_of_uav)
    Record = load_from_csv(path, data_type=float)
    
    X_feature_file = 'TESTING_NumOfUser:{}_NumOfUAV:{}_feature.csv'.format(number_of_user, number_of_uav)
    feature = load_from_csv(X_feature_file, data_type=float)

    setup_seed()

    # OURS:
    model = MemoryDNN.load_model('MODEL_NumOfUser:{}_NumOfUAV:{}.pt'.format(number_of_user, number_of_uav))
    try_method(Record, compare_method(allocate_plan_NN_model), 'NN Model', model=model, input_feature=feature)
    try_method(Record, compare_method(allocate_plan_all_upload_random), 'ALL UPLOAD RANDOM (K=1)', K=1)
    try_method(Record, compare_method(allocate_plan_all_upload_random), 'ALL UPLOAD RANDOM (K=5)', K=5)
    try_method(Record, compare_method(allocate_plan_local_and_upload_random), 'BOTH LOCAL AND UPLOAD RANDOM (K=5)', K=5)
    try_method(Record, compare_method(allocate_plan_all_local), 'ALL LOCAL', K=1)