import csv
import numpy as np
import torch
import random

def load_from_csv(file_path=None, data_type=None):
    with open(file_path, mode='r') as f:
        reader = csv.reader(f, delimiter=',')
        data = []

        for row in reader:
            data.append(row)
            
        data = np.array(data, dtype=data_type)
    
    return data

def setup_seed(seed=201314):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True