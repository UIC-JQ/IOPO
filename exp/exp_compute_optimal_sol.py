import itertools
import multiprocessing
from dataclass import DataConfig
from util import convert_index_to_zero_one_sol, load_from_csv
from opt3 import whale
from tqdm import tqdm

def generate_permutations(number_of_users, number_of_uavs, data_config):
    choices = [[i for i in range(number_of_uavs + 1)] for _ in range(number_of_users)]
    policy = [convert_index_to_zero_one_sol(sol, data_config) for sol in itertools.product(*choices)]

    print('[LOG]: Total Number of {} Policy Generated.'.format(len(policy)))
    return policy

def compute_optimal_energy_cost(policy, record, data_config, ovt_penalty=2500):
    lowest_eng_cost = float('inf')

    for policy in tqdm(policy):
        _, eng_cost = whale(record, policy, data_config, PENALTY=ovt_penalty)

        # 更新最低energy cost
        lowest_eng_cost = min(eng_cost, lowest_eng_cost)

    return lowest_eng_cost


def process_data(inputs):
    record, policy, data_config = inputs
    _, e_c = whale(record, policy, data_config, PENALTY=2500)

    return e_c

if __name__ == '__main__':
    num_user, num_uav = 10, 2
    data_config = DataConfig(load_config_from_path='Config/CONFIG_NumOfUser:{}_NumOfUAV:{}.json'.format(num_user, num_uav))
    policies = generate_permutations(num_user, num_uav, data_config)

    # 准备数据
    train_record = load_from_csv('Dataset/user:{}_uav:{}/TRAINING_NumOfUser:{}_NumOfUAV:{}_record.csv'.format(num_user, num_uav, num_user, num_uav), data_type=float)[:10]
    N_of_data = len(train_record)

    # 串行
    # -----------------------------------------------------------------
    # for record in tqdm(train_record):
    #     compute_optimal_energy_cost(policies, record, data_config)

    # 并行
    # -----------------------------------------------------------------
    # 创建进程池
    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_processes)

    # 使用进程池并行处理数据，并获取返回值迭代器
    acc_min_eng_cost = 0
    for record in tqdm(train_record):
        data_list = [[record, policies[i], data_config] for i in range(len(policies))]
        results_iter = pool.map(process_data, data_list)
        result = list(results_iter)
        min_eng_cost = min(result)
        acc_min_eng_cost += min_eng_cost

    pool.close()
    pool.join()

    print('optimal energy cost: {}'.format(acc_min_eng_cost / N_of_data))
