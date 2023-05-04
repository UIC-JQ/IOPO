import numpy as np
from tqdm import tqdm

# 更快的生成随机解
def random_sample(user, uav, K=100):
    store_arr = []
    for i in range(user):
        first_one = np.zeros((user * K, uav))
        first_one_idx = np.random.randint(0, 2, size=user * K)
        first_one[first_one_idx == 1, np.random.randint(0, uav, size = np.count_nonzero(first_one_idx))] = 1
        store_arr.append(first_one)
        

    random_sols = np.concatenate(store_arr, axis=1)
    random_sols.sort(key=lambda x: whale(record, x, data_config))

    return random_sols[0]

def __normalize_feature(self, F):
    # TODO: 修改normalize方法

    F = np.array(F)

    mean = np.mean(F, axis=1)
    std = np.std(F, axis=1)

    normalized = (F - mean) / std

    return normalized

if __name__ == '__main__':
    n = 5000
    for _ in tqdm(range(5000)):
        ans = random_sample(6, 6, K=1000)