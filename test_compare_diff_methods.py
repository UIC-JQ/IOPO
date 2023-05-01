import csv
import numpy as np
from dataclass import DataConfig
from opt3 import whale
from tqdm import tqdm

def try_method(method, method_name):
    avg_eng_cost, overtime_records_ratio, avg_overtime_ratio, avg_overtime_people = method(X, data_config)
    print('--> 方法名: [%s]', method_name)
    print('平均每条record的energy cost:', avg_eng_cost)
    print('所有record中，存在超时user的record比例:',overtime_records_ratio)
    print('所有存在超时user的record中，平均超时人数占总人数的比例:', avg_overtime_ratio)
    print('所有存在超时user的record中，平均超时人数为', avg_overtime_people)

def cmp_method_all_local_compute(X, data_config: DataConfig):
    total_eng_cost = 0
    __allocate_plan = np.zeros(X[0].shape)
    total_overtime_records_num = 0
    store_overtime_ratio = []
    store_overtime_people = []

    for record in tqdm(X):
        _, energy, overtime_logs = whale(record, __allocate_plan, data_config, need_stats=True)

        if overtime_logs:
        # 统计超时数据stat
        # 如果这条record有超时
            total_overtime_records_num += 1
            store_overtime_ratio.append(len(overtime_logs) / data_config.user_number)
            store_overtime_people.append(len(overtime_logs))

        total_eng_cost += energy
    
    avg_eng_cost                    = total_eng_cost / len(X)
    overtime_records_ratio          = total_overtime_records_num / len(X)
    avg_overtime_ratio              = sum(store_overtime_ratio) / len(store_overtime_ratio)
    avg_overtime_people             = sum(store_overtime_people) / len(store_overtime_people)

    return avg_eng_cost, overtime_records_ratio, avg_overtime_ratio, avg_overtime_people


def cmp_method_all_random(X, data_config: DataConfig):
    total_eng_cost = 0
    total_overtime_records_num = 0
    store_overtime_ratio = []
    store_overtime_people = []

    # TODO：modify here
    # RANDOM_TIMES_PER_RECORD = 

    for record in tqdm(X):
        __allocate_plan = np.zeros(X[0].shape)
        _, energy, overtime_logs = whale(record, __allocate_plan, data_config, need_stats=True)

        if overtime_logs:
        # 统计超时数据stat
        # 如果这条record有超时
            total_overtime_records_num += 1
            store_overtime_ratio.append(len(overtime_logs) / data_config.user_number)
            store_overtime_people.append(len(overtime_logs))

        total_eng_cost += energy
    
    avg_eng_cost                    = total_eng_cost / len(X)
    overtime_records_ratio          = total_overtime_records_num / len(X)
    avg_overtime_ratio              = sum(store_overtime_ratio) / len(store_overtime_ratio)
    avg_overtime_people             = sum(store_overtime_people) / len(store_overtime_people)

    return avg_eng_cost, overtime_records_ratio, avg_overtime_ratio, avg_overtime_people


if __name__ == '__main__':
    # SETTINGS:
    n_of_users = 3
    n_of_uavs  = 6
    data_config = DataConfig(n_of_user=n_of_users, n_of_uav=n_of_uavs)

    # LOAD Test data:
    path = './test_data/testData_userNumber=%d_n=5000.csv' % n_of_users

    with open(path, mode='r') as file:
        # 创建CSV读取器，指定分隔符为逗号
        reader = csv.reader(file, delimiter=',')
        # 读取CSV文件的数据到一个列表中
        X = []
        for row in reader:
            X.append(row)
        X = np.array(X, dtype=float)
    
    try_method(cmp_method_all_local_compute, 'ALL LOCAL COMPUTE')
    
    
