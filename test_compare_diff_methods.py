import csv
import numpy as np
from dataclass import DataConfig
from opt3 import whale
from tqdm import tqdm

def try_method(X, method, method_name, **kws):
    avg_eng_cost, overtime_records_ratio, avg_overtime_ratio, avg_overtime_people = method(X, data_config, **kws)
    print('--> 方法名: [%s]' % method_name)
    print('平均每条record的energy cost:', avg_eng_cost)
    print('所有record中，存在超时user的record比例: {}%'.format(overtime_records_ratio * 100))
    print('所有存在超时user的record中，平均超时人数占总人数的比例: {}%'.format(avg_overtime_ratio * 100))
    print('所有存在超时user的record中，平均超时人数为', avg_overtime_people)

def allocate_plan_all_local(record, data_config, **kws):
    return np.zeros(data_config.user_number * data_config.uav_number)

def allocate_plan_all_upload_random(record, data_config, **kws):
    _, __allocate_plan = random_sample_lower_eng_cost_plan(record,
                                                           data_config,
                                                           K=kws['K'],
                                                           exclude_local_options=True)

    return __allocate_plan

def allocate_plan_local_and_upload_random(record, data_config, **kws):
    _, __allocate_plan = random_sample_lower_eng_cost_plan(record,
                                                           data_config,
                                                           K=kws['K'],
                                                           exclude_local_options=False)
    return __allocate_plan


def compare_method(f):
    def inner(X, data_config, **kws):
        total_eng_cost = 0
        total_overtime_records_num = 0
        store_overtime_ratio = []
        store_overtime_people = []

        for record in tqdm(X):
            # 用给定方法生成一个 allocate plan
            __allocate_plan = f(record, data_config, **kws)
            # 计算energy cost
            _, energy, overtime_logs = whale(record, __allocate_plan, data_config, need_stats=True)

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

def random_sample_lower_eng_cost_plan(record, data_config, K=100, exclude_local_options=False):
    """
    date: 2023/4/30
    author: Yu, Jianqiu
    """
    config_sol_size = data_config.user_number * data_config.uav_number
    config_random_times = K
        
    e_best = float('inf')
    lower_b = 0 if not exclude_local_options else 1
    higher_b = data_config.uav_number + 1

    for _ in range(config_random_times):
        random_sol = np.zeros(config_sol_size, dtype=int)
        
        for i in range(0, config_sol_size, data_config.uav_number):
            idx = np.random.randint(lower_b, higher_b)
            if idx == 0:
                continue
            random_sol[i + idx - 1] = 1
        
        _, new_energy = whale(record, random_sol, data_config) # 重新算结果

        if new_energy < e_best:
            e_best = new_energy
            sol_ = random_sol

    return e_best, sol_

if __name__ == '__main__':
    # SETTINGS:
    n_of_users = 3
    n_of_uavs  = 6
    data_config = DataConfig(n_of_user=n_of_users, n_of_uav=n_of_uavs)

    # LOAD Test data:
    path = './test_data/testData_userNumber=%d_n=5000.csv' % n_of_users

    n = 10
    with open(path, mode='r') as file:
        # 创建CSV读取器，指定分隔符为逗号
        reader = csv.reader(file, delimiter=',')
        # 读取CSV文件的数据到一个列表中
        X = []
        for row in reader:
            if n <= 0: break
            X.append(row)
            n -= 1
        X = np.array(X, dtype=float)
    
    try_method(X, compare_method(allocate_plan_all_upload_random), 'ALL UPLOAD RANDOM (K=1)', K=1)
    try_method(X, compare_method(allocate_plan_all_upload_random), 'ALL UPLOAD RANDOM (K=10)', K=10)
    try_method(X, compare_method(allocate_plan_local_and_upload_random), 'BOTH LOCAL AND UPLOAD RANDOM (K=10)', K=10)
    try_method(X, compare_method(allocate_plan_all_local), 'ALL LOCAL', K=1)