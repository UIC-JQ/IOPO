import numpy as np
from dataclass import DataConfig
from opt3 import whale
from tqdm import tqdm
import argparse

from Model_LSTM import LSTM_Model
from Model_MLP import MLP
from Model_LSTM_IMP import Model_LSTM_IMP
from util import load_from_csv, setup_seed, generate_better_allocate_plan_KMN, convert_index_to_zero_one_sol


def try_method(X, method, method_name, **kws):
    """
    测试模型method, 并统计相关结果
    """
    avg_eng_cost, avg_ovt_penalized_eng_cost, overtime_records_ratio, avg_overtime_ratio, avg_overtime_people = method(X, data_config, **kws)
    print('--> 方法名: [%s]' % method_name)
    print('*平均每条record的energy cost (包含超时penality):', avg_ovt_penalized_eng_cost)
    print('----平均每条record的energy cost (不包含超时penality):', avg_eng_cost)
    print('*所有record中，存在超时user的record比例: {:.3f}%'.format(overtime_records_ratio * 100))
    print('*所有存在超时user的record中，平均超时人数占总人数的比例: {:.3f}%'.format(avg_overtime_ratio * 100))
    print('*所有存在超时user的record中，平均超时人数为', avg_overtime_people)

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
    _, _, __allocate_plan = model.generate_answer(X, data_config)

    if kws['print_plan']:
        print_plan(__allocate_plan, data_config)

    return __allocate_plan

def allocate_plan_NN_model_optimize(record, data_config: DataConfig, **kws):
    model = kws['model']
    data_idx = kws['data_idx']
    X = kws['input_feature'][data_idx,:]
    KMN_K = kws['KMN_K']
    prob, ans, __allocate_plan = model.generate_answer(X, data_config)

    # 计算当前 NN预测的 allocation 的 energy cost
    _, eng_cost = whale(record, __allocate_plan, data_config, PENALTY=kws['OVT_PENALTY'])

    # 尝试生成更好的解
    opt_eng_cost, new_y = generate_better_allocate_plan_KMN(ans,
                                                            __allocate_plan,
                                                            prob,
                                                            K=KMN_K,
                                                            eng_compute_func=whale,
                                                            record=record,
                                                            data_config=data_config,
                                                            convert_output_size=data_config.user_number*data_config.uav_number,
                                                            threshold_p=1 / (data_config.uav_number + 1),
                                                            PENALTY=kws['OVT_PENALTY'])

    if opt_eng_cost < eng_cost:
        __allocate_plan = convert_index_to_zero_one_sol(new_y, data_config)

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

def allocate_plan_with_time_constraint(record, data_config: DataConfig, **kws):
    data_idx = kws['data_idx']
    local_time_ranking = [eval(s) for s in kws['ranking'][data_idx]]
    user_to_uav_infos_RAW = [eval(s) for s in kws['uav_to_user_info'][data_idx]]
    user_to_uav_infos = dict()
    for items in user_to_uav_infos_RAW:
        user_idx = items[0]
        user_to_uav_infos[user_idx] = []

        for chs in items[1:][0]:
            user_to_uav_infos[user_idx].append(chs)


    _, _, __allocate_plan = data_config.allocate_with_no_overtime_constraint(record, local_time_ranking, user_to_uav_infos)

    if kws['print_plan']:
        print_plan(__allocate_plan, data_config)

    return __allocate_plan

def compare_method(allocate_plan_generation_method, PENALTY):
    def inner(X, data_config, **kws):
        total_eng_cost = 0                      # 总的energy cost
        total_overtime_penalized_eng_cost = 0   # 包含超时penalty的energy cost
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
                total_overtime_penalized_eng_cost += (PENALTY) * len(overtime_logs)

            total_eng_cost += energy
            total_overtime_penalized_eng_cost += energy

        # 计算stats: 
        avg_eng_cost                    = total_eng_cost / len(X)
        avg_overtime_pe_eng_cost        = total_overtime_penalized_eng_cost / len(X)
        overtime_records_ratio          = total_overtime_records_num / len(X)
        avg_overtime_ratio              = sum(store_overtime_ratio) / len(store_overtime_ratio) if len(store_overtime_ratio) else 0
        avg_overtime_people             = sum(store_overtime_people) / len(store_overtime_people) if len(store_overtime_people) else 0

        return avg_eng_cost, avg_overtime_pe_eng_cost, overtime_records_ratio, avg_overtime_ratio, avg_overtime_people
    
    return inner


if __name__ == '__main__':
    # 读取脚本中的 SETTINGS:
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
    data_config.overtime_penalty = 0                        # 显示energy cost的时候，去掉overtime penalty.
    OVERTIME_PENALTY = 100
    KNM_K = data_config.uav_number * data_config.user_number
    print('[Test Config] OVERTIME PENALTY is set to >>>100<<<')

    # ---------------------------------------------------------------------------------------------------------
    # LOAD Test data:
    dataset_save_dir = "user:{}_uav:{}".format(number_of_user, number_of_uav)
    path = './Dataset/{}/TESTING_NumOfUser:{}_NumOfUAV:{}_record.csv'.format(dataset_save_dir, number_of_user, number_of_uav)
    Record = load_from_csv(path, data_type=float)
    
    X_feature_file = './Dataset/{}/TESTING_NumOfUser:{}_NumOfUAV:{}_feature.csv'.format(dataset_save_dir, number_of_user, number_of_uav)
    feature = load_from_csv(X_feature_file, data_type=float)

    local_rank_file = './Dataset/{}/TESTING_NumOfUser:{}_NumOfUAV:{}_local_comp_time_ranking.csv'.format(dataset_save_dir, number_of_user, number_of_uav)
    local_time_rankings = load_from_csv(local_rank_file)

    user_uav_infos_file = './Dataset/{}/TESTING_NumOfUser:{}_NumOfUAV:{}_user_to_uav_infos.csv'.format(dataset_save_dir, number_of_user, number_of_uav)
    user_uav_infos = load_from_csv(user_uav_infos_file)

    # ---------------------------------------------------------------------------------------------------------
    setup_seed()                     # 设置固定的随机种子

    # ---------------------------------------------------------------------------------------------------------
    # 选择NN模型
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

    save_dir = './Saved_model/user:{}_uav:{}/'.format(number_of_user, number_of_uav)
    model_save_path = save_dir + 'MODEL_{}_NumOfUser:{}_NumOfUAV:{}.pt'.format(model_name, number_of_user, number_of_uav)
    model = model.load_model(model_save_path)

    # ---------------------------------------------------------------------------------------------------------
    # 开始测试不同方法
    # 1. NN model
    try_method(Record, 
               method=compare_method(allocate_plan_NN_model, OVERTIME_PENALTY),
               method_name='NN Model : {}'.format(str(model_name)),
               model=model,
               input_feature=feature,
               print_plan=False)

    try_method(Record, 
               method=compare_method(allocate_plan_NN_model_optimize, OVERTIME_PENALTY),
               method_name='NN Model : {} (better solution is generated during test with KNM algorithm, K={})'.format(str(model_name), KNM_K),
               model=model,
               input_feature=feature,
               KMN_K=KNM_K,
               OVT_PENALTY=OVERTIME_PENALTY,
               print_plan=False)
    
    if args.test_NN_only:
        print('[Config] Only test NN method, system exit.')
        exit(0)

    print('-' * 50)
    # ALL LOCAL
    try_method(Record, compare_method(allocate_plan_all_local, OVERTIME_PENALTY), 'ALL LOCAL', K=1, print_plan=False)

    print('-' * 50)
    # Greedy (without time constraint)
    try_method(Record,
               method=compare_method(allocate_plan_by_local_compute_t_and_uav_comp_speed, OVERTIME_PENALTY),
               method_name='Greedy (by Local Compute time and uav Compute speed)',
               ranking=local_time_rankings,
               print_plan=False)

    print('-' * 50)
    # Greedy (with time constraint)
    try_method(Record,
               method=compare_method(allocate_plan_with_time_constraint, OVERTIME_PENALTY),
               method_name='Greedy (with no-overtime constraint)',
               ranking=local_time_rankings,
               uav_to_user_info=user_uav_infos,
               print_plan=False)

    print('-' * 50)
    # random (without time constraint), randomly select from upload choices only
    try_method(Record, compare_method(allocate_plan_all_upload_random, OVERTIME_PENALTY), 'ALL UPLOAD OPTIMIZED RANDOM (K=1)', K=1, print_plan=False)

    print('-' * 50)
    # random (without time constraint), randomly select from upload + local choices
    try_method(Record, compare_method(allocate_plan_local_and_upload_random, OVERTIME_PENALTY), '(LOCAL + UPLOAD) OPTIMIZED RANDOM (K=1)', K=1, print_plan=False)