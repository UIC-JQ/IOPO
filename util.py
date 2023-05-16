import csv
import numpy as np
import torch
import random
import torch.nn as nn
import torch
import os

def load_from_csv(file_path=None, data_type=None):
    with open(file_path, mode='r') as f:
        reader = csv.reader(f, delimiter=',')
        data = []

        for row in reader:
            data.append(row)

        if data_type != None: 
            data = np.array(data, dtype=data_type)
    
    return data

def build_dir(path_to_dir):
    os.makedirs(path_to_dir, exist_ok=True)

def setup_seed(seed=201314):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
    
def convert_index_to_zero_one_sol(plan, data_config):
    zero_one_sol = np.zeros(data_config.user_number * data_config.uav_number)
    base = 0

    for i in range(data_config.user_number):
        ans_idx = int(plan[i])
        if ans_idx != 0:
            zero_one_sol[base + ans_idx - 1] = 1

        base += data_config.uav_number
    
    return zero_one_sol

def generate_better_allocate_plan_KMN(prob,
                                      K=1, 
                                      threshold_p=0.5,
                                      eng_compute_func=None,
                                      record_idx=None,
                                      convert_output_size=None,
                                      data_config=None,
                                      record=None,
                                      PENALTY=None):
    flatten_prob = prob.view(-1, 1)
    # Try use average as threshold
    # threshold_p = float(torch.mean(flatten_prob.view(1, -1)))
    # print(threshold_p)
    CUTOFF_PROB = threshold_p

    threshold_0 = torch.full(flatten_prob.shape, fill_value=threshold_p, dtype=torch.float32)

    # DIST = abs(Euclidean Distance(p - threshold_p)), 排序
    _, idx_list = torch.sort(nn.functional.pairwise_distance(threshold_0, flatten_prob, p=2))

    # 记录最好的answer
    eng_cost, final_allocation_plan = float('inf'), None

    # 生成K个新解
    for i in range(K):
        new_sol = np.zeros(convert_output_size)
        new_sol_tensor = []

        # 选择新的threshold, DIST中第i大的prob, 作为新的threshold
        idx = idx_list[i]
        threshold = flatten_prob[idx]

        # 规则：
        # prob > threshold -> 1
        # prob = threshold and threshold <= CUTOFF_PROB -> 1
        # prob = threshold and threshold > CUTOFF_PROB -> 0
        # prob < threshold -> 0
        for ii, row in enumerate(prob):
            selected = False

            for ij, prob_compare in enumerate(row):
                if prob_compare > threshold:
                    # 本地满足要求:
                    if ij == 0: 
                        new_sol_tensor.append(0)
                        selected = True
                        break
                    # 计算无人机编号:
                    new_sol[ii * data_config.uav_number + ij - 1] = 1
                    new_sol_tensor.append(ij)
                    selected = True
                    break
                elif prob_compare == threshold:
                    if threshold > CUTOFF_PROB:
                        continue
                    # 本地满足要求:
                    if ij == 0: 
                        new_sol_tensor.append(0)
                        selected = True
                        break
                    # 计算无人机编号:
                    new_sol[ii * data_config.uav_number + ij - 1] = 1
                    new_sol_tensor.append(ij)
                    selected = True
                    break
            
            if not selected:
                new_sol_tensor.append(0)
        if record_idx is not None: 
            _, new_energy_cost = eng_compute_func(record_idx, new_sol)
        else:
            _, new_energy_cost = eng_compute_func(record, new_sol, data_config, PENALTY=PENALTY)
        # 更新answer:
        if new_energy_cost < eng_cost:
            eng_cost = new_energy_cost
            final_allocation_plan = new_sol_tensor

    return eng_cost, torch.Tensor(final_allocation_plan)