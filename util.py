import csv
import numpy as np
import torch
import random
import torch.nn as nn

def load_from_csv(file_path=None, data_type=None):
    with open(file_path, mode='r') as f:
        reader = csv.reader(f, delimiter=',')
        data = []

        for row in reader:
            data.append(row)

        if data_type != None: 
            data = np.array(data, dtype=data_type)
    
    return data

def setup_seed(seed=201314):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def generate_better_allocate_plan(self, prob, K=1, threshold_v=0.5, eng_compute_func=None, idx=None):
    flatten_prob = prob.view(-1, 1)

    threshold_0 = torch.full(flatten_prob.shape, fill_value=threshold_v, dtype=torch.float32)

    # 将概率排序大小
    _, idx_list = torch.sort(nn.functional.pairwise_distance(threshold_0, flatten_prob, p=2))

    eng_cost, final_allocation_plan = float('inf'), None
    CUTOFF_PROB = 0.5

    # 生成K-1个新解
    for i in range(K-1):
        new_sol = np.zeros(self.convert_output_size)
        new_sol_tensor = []

        idx = idx_list[i]
        threshold = flatten_prob[idx]

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
                    new_sol[ii * self.data_config.uav_number + ij - 1] = 1
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
                    new_sol[ii * self.data_config.uav_number + ij - 1] = 1
                    new_sol_tensor.append(ij)
                    selected = True
                    break
            
            if not selected:
                new_sol_tensor.append(0)
        
        _, new_energy_cost = eng_compute_func(idx, new_sol)
        if new_energy_cost < eng_cost:
            eng_cost = new_energy_cost
            final_allocation_plan = new_sol_tensor

        # > threshold -> 1
        # = threshold and threshold <= 0.5 -> 1
        # = threshold and threshold > 0.5 -> 0
        # < threshold -> 0
    
    return eng_cost, torch.Tensor(final_allocation_plan)