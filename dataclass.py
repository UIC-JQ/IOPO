import numpy as np
import csv
from tqdm import tqdm
import json
from opt3 import whale

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
        self.SUB_CHANNEL_K                  = 0.00001          # 每个子频道中央频率的吸收参数 db/m
        self.CHANNEL_BANDWIDTH              = 3e10             # Hz

        # ---------------------------------------
        # modify following variables (every test cases)
        self.overtime_penalty               = penalty                # overtime penalty
        self.user_number                    = n_of_user              # number of users
        self.uav_number                     = n_of_uav               # number of uavs

        # 板子参数：
        self.IRS_delta                = 0.05                      # IRS width 板子宽度 (unit: meter)
        self.IRS_l0_coordinate        = np.array((20, 0, 20))     # IRS第一块坐标
        self.IRS_x_number             = 5#IRS x-axis refector
        self.IRS_z_number             = 5#IRS z-axis refector

        # the computation rate of users (local) UEDs cycles/slot
        __USER_C_LOWER_BOUND        = 5000
        __USER_C_HIGHER_BOUND       = 20000
        self.user_computational_capacity = [np.random.randint(__USER_C_LOWER_BOUND, __USER_C_HIGHER_BOUND)
                                                 for _ in range(self.user_number)]    
        # 用户计算功率 j/slot
        __USER_C_POWER_LOWER_B      = 0.01
        __USER_C_POWER_HIGHER_B     = 0.05
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
        self.__UAV_COMP_CAP_LOWER_B      = 80000
        self.__UAV_COMP_CAP_HIGHER_B     = 125000
        self.uav_computational_capacity = [np.random.randint(self.__UAV_COMP_CAP_LOWER_B, self.__UAV_COMP_CAP_HIGHER_B)
                                                for _ in range(self.uav_number)]
        # 2.无人机功率 j/slot = 600W
        __UAV_POWER_LOW_B            = 0.05
        __UAV_POWER_HIGH_B           = 0.08
        self.uav_computational_power    = [np.random.uniform(__UAV_POWER_LOW_B, __UAV_POWER_HIGH_B)
                                                for _ in range(self.uav_number)]
        # 3.无人机位置
        # format：(x, y, height)
        # 只有4，6两种情况
        __uav_coordinate = {
            4: [],
            6: np.array([[61, 321, 20], 
                [569, 383, 20], 
                [415, 208, 20], 
                [203, 80, 20], 
                [207, 211, 20], 
                [190, 78, 20],])
        }
        if self.uav_number != 6:
            raise NotImplementedError()
        
        self.uav_coordinate = __uav_coordinate[self.uav_number]

        # -----------------------------------------------------------------------
        # 用于生成数据集的设置
        self.dataset_board_x_size = 600
        self.dataset_board_y_size = 800

        # 单位：bit
        self.dataset_user_task_size_l_b = 300 * 1000    # 300KB
        self.dataset_user_task_size_h_b = 500 * 1000    # 500KB

        # 单位: cycle/bit
        self.dataset_cpb_l_b = 40
        self.dataset_cpb_h_b = 150
        
        # 单位: slot (1.25ms)
        # 超过本地计算需要时间多久是可以的
        self.dataset_dataset_cut_off_time = 100

        # 生成每个数据的标准答案，需要的random次数
        self.dataset_random_times_for_selecting_best_sol = 100

        # -------------------------------------------------------------------------
        # 用于鲸鱼算法优化
        self.optimize_num_whales = 3
        self.optimize_max_iter = 5

        
    def generate_dataset(self, num_of_data_points=5000, saving_path=None, K=None):
        # format: (user1_x, user1_y, user_1_z, user2_x, user_2_y, user2_z, ..., user_1_task_size, user_1_cpu_needed, user_1_tolerance)
        data_for_computing_energy_cost = [] 

        # format: (f1, f2, ..., fn)
        # fi = (distance_to_uavs (用户距离无人机的距离) size=uav_number, 
        #       task_package_size (分配给用户的task包的大小, 影响传输速度) size = 1                  [x]
        #       user_transfer_power (用户传输能量消耗功率) size = 1                                [x]
        #       task_finish_time_threshold (允许完成任务的时间(不超过这个时间都可以)) size = 1       [x]
        #       time_local_time_need (本地完成任务所需要的时间) size = 1                           [x]
        #       eng_cost_user_loacl_compute（用户本地计算的能量消耗) size = 1                      [x]
        #       time_each_uav_finish_task (在每个无人机上完成task需要的时间) size=uav_number        [x]
        #       eng_cost_uavs_compute (每个无人机上完成task需要消耗的能量) size=uav_number          [x]
        #       )
        # size_of_fi = (uav_number * 3 + 1 * 5)
        # size_of x_i = f_i * user_number
        data_X_features = []

        data_Y = []
        K = self.dataset_random_times_for_selecting_best_sol if not K else K

        data_eng_cost = []

        for _ in tqdm(range(num_of_data_points)):
            eng_cost = []
            feature = []

            # 生成用户坐标：(x, y, z)
            for i in range(self.user_number):
                x = np.random.uniform(0, self.dataset_board_x_size)
                y = np.random.uniform(0, self.dataset_board_y_size)
                eng_cost.extend([x, y, 0])

                # FEATURE: distance_to_uavs (用户距离无人机的距离)
                for idx in range(self.uav_number):
                    # 用户i对无人机j的距离
                    dist = np.linalg.norm(np.array([x, y, 0]) - self.uav_coordinate[idx])
                    # print('point: {}, uav co: {}, distance to uav: {} is: {}'.format((x,y,0), self.uav_coordinate[idx], idx, dist))
                    feature.append(dist)
            
            # 生成task info
            for i in range(self.user_number):
                #用户平均任务大小 Mbit/slot单位时间多少Mbit
                task_size = np.random.randint(self.dataset_user_task_size_l_b, self.dataset_user_task_size_h_b) 
                eng_cost.append(task_size)

                cpb = np.random.randint(self.dataset_cpb_l_b, self.dataset_cpb_h_b)
                cpu_cycles_need = task_size * cpb #任务需要的cpu数量
                # 用于whale的计算
                eng_cost.append(cpu_cycles_need)

                # time_requirement_l_b = np.ceil( (cpu_cycles_need) / self.__UAV_COMP_CAP_HIGHER_B ) # TODO: FIX ME 也许是__UAV_COMP_CAP_LOWER_B
                time_requirement_l_b = np.ceil( (cpu_cycles_need) / self.__UAV_COMP_CAP_LOWER_B )
                __local_compute_time = (cpu_cycles_need) / self.user_computational_capacity[i]
                time_requirement_h_b = np.ceil(__local_compute_time) 

                assert time_requirement_l_b < time_requirement_h_b
                compute_time_allowed = np.random.randint(time_requirement_l_b, time_requirement_h_b) + self.dataset_dataset_cut_off_time
                eng_cost.append(compute_time_allowed) # 任务可以接受的延迟 slot

                # ----------------------------------------------------------------
                # 生成Feature:

                # FEATURE: task_package_size (分配给用户的task包的大小, 影响传输速度) 
                feature.append(task_size)

                # FEATURE: user_transfer_power (用户传输能量消耗功率)
                feature.append(self.user_transmit_power[i])

                # FEATURE: task_finish_time_threshold (允许完成任务的时间(不超过这个时间都可以))
                feature.append(compute_time_allowed)

                # FEATURE: time_local_time_need (本地完成任务所需要的时间)
                feature.append(__local_compute_time)

                # FEATURE: eng_cost_user_loacl_compute（用户本地计算的能量消耗)
                feature.append(__local_compute_time * self.user_computation_power[i])

                for idx in range(self.uav_number):
                    # FEATURE: time_each_uav_finish_task (在每个无人机上完成task需要的时间)
                    t = cpu_cycles_need / self.uav_computational_capacity[idx]
                    feature.append(t)

                    # print(compute_time_allowed, __local_compute_time, t)

                    # FEATURE: eng_cost_uavs_compute (每个无人机上完成task需要消耗的能量)
                    feature.append(t * self.uav_computational_power[idx])

            data_for_computing_energy_cost.append(eng_cost)
            data_X_features.append(feature)

            # 生成当前record的解 (由0, 1构成，每个user最多有一个1)
            e_cost, sol = self.random_sample_lower_eng_cost_plan(eng_cost, K=K, exclude_local_options=False)

            # 将解转换成编号
            # [0, 0, 1] -> 选择无人机编号3
            # [0, 0, 0] -> 0, 代表本地计算
            Y = []
            for dd in range(self.user_number):
                base = dd * self.uav_number
                find_one_idx = np.where(sol[base: base + self.uav_number])[0]

                if len(find_one_idx) < 1:
                    Y.append(0)
                else:
                    Y.append(find_one_idx[0] + 1)
                
            # print(Y, sol)
            data_Y.append(np.array(Y))
            data_eng_cost.append([e_cost])

        self.__save_to_csv(data_for_computing_energy_cost, saving_path + '_record')
        self.__save_to_csv(data_X_features, saving_path + '_feature')
        self.__save_to_csv(data_Y, saving_path + '_solution')
        self.__save_to_csv(data_eng_cost, saving_path + '_energy_cost')
    
    def __save_to_csv(self, data, file_name, file_type='.csv'):
        with open(file_name + file_type, mode='w+', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(data)


    def load_config_from_file(self, file_path, display=False):
        """
        作用：从配置文件file_path中读取配置
        """

        print('---> loading config from file: "{}"'.format(file_path))
        self.__dict__.clear()

        with open(file_path, 'r') as f:
            for k, v in json.load(f).items():
                # 将所有list转换为ndarray类型
                if type(v) == list:
                    self.__dict__[k] = np.array(v)
                    continue

                self.__dict__[k] = v

        print('Done.')
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

    def random_sample_lower_eng_cost_plan(self, record, K=100, exclude_local_options=False):
        """
        date: 2023/4/30
        author: Yu, Jianqiu
        """
        config_sol_size = self.user_number * self.uav_number
        config_random_times = K
            
        e_best = float('inf')
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
    # 创建对象
    number_of_user = 3
    number_of_uav = 6
    dataObj = DataConfig(n_of_user=number_of_user, n_of_uav=number_of_uav)
    
    # 保存config
    dataObj.save_config('CONFIG_NumOfUser:{}_NumOfUAV:{}.json'.format(number_of_user, number_of_uav))

    # 生成数据集:
    dataObj.generate_dataset(num_of_data_points=10000, saving_path='./TRAINING_NumOfUser:{}_NumOfUAV:{}'.format(number_of_user, number_of_uav), K=100)
    dataObj.generate_dataset(num_of_data_points=5000, saving_path='./TESTING_NumOfUser:{}_NumOfUAV:{}'.format(number_of_user, number_of_uav), K=1)
