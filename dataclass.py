import csv
import json
import heapq
import numpy as np
import argparse
from tqdm import tqdm
from opt3 import whale, compute_local_eng_cost, compute_upload_eng_cost

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class DataConfig:
    def __init__(self, 
                 n_of_uav=None,
                 n_of_user=None,
                 penalty=1000,
                 load_config_from_path=None) -> None:
        
        if load_config_from_path is not None:
            self.load_config_from_file(load_config_from_path)
            return

        assert n_of_user is not None and n_of_user is not None
        # CONSTANT VALUES:
        self.SPEED_OF_LIGHT                 = 299792458        # speed of the light m/s
        self.TIME_SLOT_LENGTH               = 0.125 * 1e-3     # s equal to 0.125ms,time of one slot
        self.CHANNEL_POWER                  = 2                # 信道传输功率
        self.SUB_CHANNEL_FREQUENCY          = 3.7e11           # 子频道的中央频率 Hz
        # self.__SUB_CHANNEL_K_L              = 0.0000001
        # self.__SUB_CHANNEL_K_H              = 0.0008
        self.SUB_CHANNEL_K                  = 0.0000001          # 每个子频道中央频率的吸收参数 db/m
        self.CHANNEL_BANDWIDTH              = 3e10             # Hz

        # ---------------------------------------
        # modify following variables (every test cases)
        self.overtime_penalty               = penalty                # overtime penalty
        self.user_number                    = n_of_user              # number of users
        self.uav_number                     = n_of_uav               # number of uavs

        # 板子参数：
        self.IRS_delta                = 0.05                      # IRS width 板子宽度 (unit: meter)
        self.IRS_l0_coordinate        = np.array((20, 0, 20))     # IRS第一块坐标
        self.IRS_x_number             = 5 #IRS x-axis refector
        self.IRS_z_number             = 5 #IRS z-axis refector

        # the computation rate of users (local) UEDs cycles/slot
        __USER_C_LOWER_BOUND        = 10000
        __USER_C_HIGHER_BOUND       = 20000
        self.user_computational_capacity = [np.random.randint(__USER_C_LOWER_BOUND, __USER_C_HIGHER_BOUND)
                                                 for _ in range(self.user_number)]    
        # 用户计算功率 j/slot
        __USER_C_POWER_LOWER_B      = 10
        __USER_C_POWER_HIGHER_B     = 30
        self.user_computation_power    = [np.random.uniform(__USER_C_POWER_LOWER_B, __USER_C_POWER_HIGHER_B)
                                                 for _ in range(self.user_number)]
        
        # 用户传输功率 j/slot
        __USER_T_POWER_LOWER_B      = 0.001
        __USER_T_POWER_HIGHER_B     = 0.005
        self.user_transmit_power    = [np.random.uniform(__USER_T_POWER_LOWER_B, __USER_T_POWER_HIGHER_B)
                                                for _ in range(self.user_number)]

        # -----------------------------------------------------------------------
        # 无人机:
        # 1. the computation rate of UAVs cycles/slot
        self.__UAV_COMP_CAP_LOWER_B      = 50000
        self.__UAV_COMP_CAP_HIGHER_B     = 60000
        self.uav_computational_capacity = [np.random.randint(self.__UAV_COMP_CAP_LOWER_B, self.__UAV_COMP_CAP_HIGHER_B)
                                                for _ in range(self.uav_number)]
        self.uav_computational_capacity = [
            15000, 20000, 30000
        ]
        # 2.无人机功率 j/slot
        __UAV_POWER_LOW_B            = 0.05
        __UAV_POWER_HIGH_B           = 0.08
        self.uav_computational_power    = [np.random.uniform(__UAV_POWER_LOW_B, __UAV_POWER_HIGH_B)
                                                for _ in range(self.uav_number)]
        # 3.无人机位置
        # format：(x, y, height)
        # __uav_coordinate = {
        #     6: np.array([[0, 0, 20], 
        #         [0, 800, 20], 
        #         [600, 800, 20], 
        #         [600, 0, 20], 
        #         [300, 400, 20], 
        #         [190, 78, 20],])
        # }

        # -----------------------------------------------------------------------
        # 用于生成数据集的设置
        self.dataset_board_x_size = 600
        self.dataset_board_y_size = 800

        # Date 5/9: 使用随机生成无人机位置
        __UAV_POS_X = np.random.uniform(0, self.dataset_board_x_size, size=(self.uav_number, 1))
        __UAV_POS_Y = np.random.uniform(0, self.dataset_board_y_size, size=(self.uav_number, 1))
        __UAV_POS_Z = np.full(shape=(self.uav_number, 1), fill_value=20)
        self.uav_coordinate = np.concatenate([__UAV_POS_X, __UAV_POS_Y, __UAV_POS_Z], axis=1)

        # 单位：bit
        self.dataset_user_task_size_l_b = 1  * 1000         # kilo bit
        # self.dataset_user_task_size_h_b = 500 * 1000          # Kilo bit
        self.dataset_user_task_size_h_b = 3 * 1000          # Kilo bit

        # 单位: needed cpu cycles / bit
        self.dataset_cpb_l_b = 149
        # self.dataset_cpb_l_b = 40
        self.dataset_cpb_h_b = 150
        
        # 超过本地计算需要时间多久是可以的
        # 单位: slot (1.25ms)
        self.dataset_dataset_cut_off_time = 0

        # 默认生成每个数据的标准答案，需要的random次数
        self.dataset_random_times_for_selecting_best_sol = 100

        # -------------------------------------------------------------------------
        # 用于鲸鱼算法优化
        self.optimize_num_whales = 3
        self.optimize_max_iter = 5

        
    def generate_dataset(self, 
                         num_of_data_points=5000,
                         saving_path=None,
                         K=None,
                         data_config=None,
                         require_feature_norm=True,
                         exclude_local_options_when_generating_y=False,
                         generate_answer_using_random_optimize=False):
        # config:
        K = self.dataset_random_times_for_selecting_best_sol if not K else K            # 生成y需要的随机次数

        # 数据存储:
        # 1. record
        # 作用：用来优化板子的phase
        # format: (user1_x, user1_y, user_1_z, user2_x, user_2_y, user2_z, ..., user_1_task_size, user_1_cpu_needed, user_1_tolerance)
        data_Records = [] 
        data_local_compute_time_rank = []

        # 2. feature:
        data_X_features = []

        # 3. Y:
        # 作用: 用来充当NN的reference answer, 用来计算loss
        data_Y = []

        # 4. energy cost:
        # 作用：记录每一个reference answer的总系统能耗
        data_eng_cost = []

        # 打印生成解的平均energy cost
        log_cummulate_eng_cost = 0

        # FEATURE1:
        # 无人机的计算速度
        feature_uav_computational_capacity = self.uav_computational_capacity


        for _ in tqdm(range(num_of_data_points)):
            record = []
            # FEATURE2:
            # 每个用户每个选择的energy cost
            feature_user_choice_eng_cost = []

            # 生成用户坐标：(x, y, z)
            user_coordinate = []

            for _ in range(self.user_number):
                x = np.random.uniform(0, self.dataset_board_x_size)
                y = np.random.uniform(0, self.dataset_board_y_size)
                record.extend([x, y, 0])
                user_coordinate.append(np.array([x, y, 0]))
            
            # 生成task record
            local_compute_time_rank = []
            for user_idx in range(self.user_number):
                task_size = np.random.randint(self.dataset_user_task_size_l_b, self.dataset_user_task_size_h_b)     # 1.用户的task大小

                cpb = np.random.randint(self.dataset_cpb_l_b, self.dataset_cpb_h_b)
                cpu_cycles_need = task_size * cpb                                                                   # 2.任务需要的cpu数量

                __acceptable_time_l_b = np.ceil( (cpu_cycles_need) / self.__UAV_COMP_CAP_LOWER_B )                  # __UAV_COMP_CAP_LOWER_B: 计算最慢的无人机的计算速度
                __local_compute_time = (cpu_cycles_need) / self.user_computational_capacity[user_idx]
                __acceptable_time_h_b = np.ceil(__local_compute_time) 

                # acceptable time
                # 1.随机生成
                # if __acceptable_time_h_b == __acceptable_time_l_b:
                #     acceptable_time = __acceptable_time_h_b + self.dataset_dataset_cut_off_time
                # else:
                #     acceptable_time = np.random.randint(__acceptable_time_l_b, __acceptable_time_h_b) + self.dataset_dataset_cut_off_time     # 任务可以接受的完成时间
                # 2. 本地的计算时间是最慢的可接受时间
                acceptable_time = __acceptable_time_h_b

                record.append(task_size)
                record.append(cpu_cycles_need)
                record.append(acceptable_time) 
                local_compute_time_rank.append((__local_compute_time, user_idx))
                # ----------------------------------------------------------------
                # 生成Feature:
                # feature2.1: 用户本地计算的energy cost
                if not exclude_local_options_when_generating_y:
                    f_user_local_eng_cost = compute_local_eng_cost(task_size=cpu_cycles_need,
                                                                   time_threshold=acceptable_time,
                                                                   user_compute_speed=self.user_computational_capacity[user_idx],
                                                                   user_compute_power=self.user_computation_power[user_idx],
                                                                   penalty=self.overtime_penalty)

                    feature_user_choice_eng_cost.append(f_user_local_eng_cost)

                # 生成Feature:
                # feature2.2: 用户上传无人机计算的energy cost
                for uav_idx in range(self.uav_number):
                    f_user_upload_eng_cost = compute_upload_eng_cost(task_size=cpu_cycles_need,
                                                                     time_threshold=acceptable_time,
                                                                     uav_compute_speed=self.uav_computational_capacity[uav_idx],
                                                                     uav_compute_power=self.uav_computational_power[uav_idx],
                                                                     penalty=self.overtime_penalty,
                                                                     uav_coordinate=self.uav_coordinate[uav_idx],
                                                                     user_coordinate=user_coordinate[user_idx],
                                                                     user_transmit_power=self.user_transmit_power[user_idx],
                                                                     data_config=data_config)

                    feature_user_choice_eng_cost.append(f_user_upload_eng_cost)

            # ----------存储数据---------------
            # 1.存储record
            data_Records.append(record)

            # 2.存储Feature
            Features = [
                feature_user_choice_eng_cost,
                feature_uav_computational_capacity
            ]

            if require_feature_norm:
                # data_X_features.append(np.concatenate([self.__normalize_feature(f) for f in Features]))
                data_X_features.append(np.concatenate([self.__normalize_feature(Features[0]), self.__divide_by_sum_nrom(Features[1])]))
            else:
                data_X_features.append(np.concatenate(Features))

            # 3.存储allocation plan:
            # 生成当前record的解 (由0, 1构成，每个user最多有一个1)
            if not generate_answer_using_random_optimize:
                # Greedy 生成策略
                local_compute_time_rank.sort(reverse=True)
                e_cost, sol, _ = self.allocate_by_local_time_and_uav_comp_speed(record, local_compute_time_rank)
                Y = sol
            else:
                # Random生成策略
                e_cost, sol = self.random_sample_lower_eng_cost_plan(record,
                                                                     K=K,
                                                                     exclude_local_options=exclude_local_options_when_generating_y)
                # 将zero-one解转换成编号
                # e.g.:
                #   1. [0, 0, 1] -> 选择无人机编号3
                #   2. [0, 0, 0] -> 0, 代表本地计算
                Y = []
                for dd in range(self.user_number):
                    base = dd * self.uav_number
                    find_one_idx = np.where(sol[base: base + self.uav_number])[0]

                    if len(find_one_idx) < 1:
                        Y.append(0)
                    else:
                        Y.append(find_one_idx[0] + 1)
                
                Y = np.array(Y)
                
            log_cummulate_eng_cost += e_cost
            data_Y.append(Y)

            # 4.存储energy cost
            data_eng_cost.append([e_cost])

            # 5.存储ranking
            data_local_compute_time_rank.append(local_compute_time_rank)
        
        print('[LOG]: average energy cost:', log_cummulate_eng_cost / num_of_data_points)
        self.__save_to_csv(data_Records, saving_path + '_record')
        self.__save_to_csv(data_X_features, saving_path + '_feature')
        self.__save_to_csv(data_Y, saving_path + '_solution')
        self.__save_to_csv(data_eng_cost, saving_path + '_energy_cost')
        self.__save_to_csv(data_local_compute_time_rank, saving_path + '_local_comp_time_ranking')
    
    def __normalize_feature(self, F):
        """
        z-score nrom
        """
        F = np.array(F)

        mean = np.mean(F)
        std = np.std(F)

        normalized = (F - mean) / std

        return normalized
    
    def __divide_by_sum_nrom(self, F):
        """
        divide by sum nrom
        """
        F = np.array(F)

        sum_of_F = sum(F)
        normalized = F / sum_of_F

        return normalized
    
    def __save_to_csv(self, data, file_name, file_type='.csv'):
        with open(file_name + file_type, mode='w+', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(data)

    def load_config_from_file(self, file_path, display=False):
        """
        作用：从配置文件file_path中读取配置
        """

        print('[config] loading config from file: "{}"'.format(file_path))
        self.__dict__.clear()

        with open(file_path, 'r') as f:
            for k, v in json.load(f).items():
                # 将所有list转换为ndarray类型
                if type(v) == list:
                    self.__dict__[k] = np.array(v)
                    continue

                self.__dict__[k] = v

        print('[config] Successed.')
        if not display:
            return

        print('Loaded configs:')
        for k, v in self.__dict__.items():
            print('* name: {}, value: {}, type: {}'.format(k, v, type(v)))
        
        
    def save_config(self, file_path):
        """
        作用：将本次使用的配置保存到文件file_path中
        """

        print('---> saving configs to file: "{}"'.format(file_path))

        with open(file_path, 'w', encoding='utf-8') as fp:
            json.dump(self.__dict__, fp, cls=NumpyEncoder)

        print('Done.')
        print('Saved configs:')
        for k, v in self.__dict__.items():
            print('* name: {}, value: {}, type: {}'.format(k, v, type(v)))
    
    def allocate_by_local_time_and_uav_comp_speed(self, record, local_t_ranking):
        allocation_plan = [0] * self.user_number
        uav_compute_speed = [(-v, i) for i, v in enumerate(self.uav_computational_capacity, start=1)]
        user_compute_speed = sorted(self.user_computational_capacity[::])
        uav_workload = [1] * self.uav_number

        heapq.heapify(uav_compute_speed)
        idx = 0
        
        while uav_compute_speed:
            _, user_idx = local_t_ranking[idx]
            # print('user index', user_idx)

            _, uav_idx = heapq.heappop(uav_compute_speed)

            temp_uav_speed = self.uav_computational_capacity[uav_idx - 1] / (uav_workload[uav_idx - 1])
            # print('selected uav: {}, speed: {}'.format(uav_idx, temp_uav_speed))

            while user_compute_speed and temp_uav_speed < user_compute_speed[-1]:
                # print('looping...')
                # print('uav_speed: {}, user_compute_speed: {}'.format(temp_uav_speed, user_compute_speed[-1]))
                user_compute_speed.pop()
        
            if not user_compute_speed:
                break
            
            # allocate
            # print('[*] user{} -> uav {}'.format(user_idx, uav_idx))
            allocation_plan[user_idx] = uav_idx

            # update info
            uav_workload[uav_idx - 1] += 1
            heapq.heappush(uav_compute_speed, (-self.uav_computational_capacity[uav_idx - 1] / uav_workload[uav_idx - 1], uav_idx))
            # print('after:', uav_compute_speed)
            idx += 1

        # 计算eng cost:
        sol_one_zero = np.zeros(self.uav_number * self.user_number)
        base = 0
        for i in range(self.user_number):
            if allocation_plan[i] == 0:
                base += self.uav_number
            else:
                sol_one_zero[base + allocation_plan[i] - 1] = 1
                base += self.uav_number
            

        _, eng_cost = whale(record, sol_one_zero, self)               # 计算greedy方法生成解的energy cost.
        
        return eng_cost, np.array(allocation_plan), sol_one_zero

    def random_sample_lower_eng_cost_plan(self, record, K=100, exclude_local_options=False):
        """
        date: 2023/4/30
        author: Yu; Jianqiu
        作用：完全随机生成一些分配方案，计算每种分配方案的energy cost, 返回energy cost最小的1条分配方案。
        """
        config_sol_size = self.user_number * self.uav_number
        config_random_times = K
            
        e_best = float('inf')
        # 0 代表本地计算
        # 如果exclude_local_options, 则排除0
        lower_b = 0 if not exclude_local_options else 1
        higher_b = self.uav_number + 1

        for _ in range(config_random_times):
            random_sol = np.zeros(config_sol_size, dtype=int)
            
            for i in range(0, config_sol_size, self.uav_number):
                idx = np.random.randint(lower_b, higher_b)
                if idx == 0:
                    continue
                random_sol[i + idx - 1] = 1
            
            _, new_energy = whale(record, random_sol, self) # 重新算结果

            if new_energy < e_best:
                e_best = new_energy
                sol_ = random_sol

        return e_best, sol_

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--uavNumber', type=int, help='uav的数量')
    parser.add_argument('--userNumber', type=int, help='user的数量')
    parser.add_argument('--number_of_train_data', type=int, help='train_data的数量')
    parser.add_argument('--number_of_test_data', type=int, help='test_data的数量')
    parser.add_argument('--using_random_sol', action='store_true', help='是否选择random方法生成解', default=False)
    args = parser.parse_args()

    # 创建对象
    number_of_user                         = args.userNumber
    number_of_uav                          = args.uavNumber
    feature_norm                           = True
    exclude_local_choice_when_generating_y = False
    generate_sol_using_random              = args.using_random_sol
    if generate_sol_using_random:
        print('[Data Generation] Using randomly generated solution')
    else:
        print('[Data Generation] Using greedly generated solution')

    dataObj                                = DataConfig(n_of_user=number_of_user, n_of_uav=number_of_uav)
    
    # 保存config
    dataObj.save_config('CONFIG_NumOfUser:{}_NumOfUAV:{}.json'.format(number_of_user, number_of_uav))

    # 生成数据集:
    dataObj.generate_dataset(num_of_data_points=args.number_of_train_data,
                             saving_path='./TRAINING_NumOfUser:{}_NumOfUAV:{}'.format(number_of_user, number_of_uav),
                             K=100,
                             data_config=dataObj,
                             require_feature_norm=feature_norm, 
                             exclude_local_options_when_generating_y=exclude_local_choice_when_generating_y,
                             generate_answer_using_random_optimize=generate_sol_using_random)

    dataObj.generate_dataset(num_of_data_points=args.number_of_test_data,
                             saving_path='./TESTING_NumOfUser:{}_NumOfUAV:{}'.format(number_of_user, number_of_uav),
                             K=1,
                             data_config=dataObj,
                             require_feature_norm=feature_norm,
                             exclude_local_options_when_generating_y=exclude_local_choice_when_generating_y,
                             generate_answer_using_random_optimize=generate_sol_using_random)
