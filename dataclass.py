import numpy as np
import csv
from tqdm import tqdm
import json

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
        self.user_computational_capacity = [np.random.randint(__USER_C_LOWER_BOUND, __USER_C_HIGHER_BOUND + 1)
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

        
        # 无人机:
        # 1. the computation rate of UAVs cycles/slot
        __UAV_COMP_CAP_LOWER_B      = 80000
        __UAV_COMP_CAP_HIGHER_B     = 125000
        self.uav_computational_capacity = [np.random.randint(__UAV_COMP_CAP_LOWER_B, __UAV_COMP_CAP_HIGHER_B)
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
            6: [61, 321, 20, 
                569, 383, 20, 
                415, 208, 20, 
                203, 80, 20, 
                207, 211, 20, 
                190, 78, 20]
        }
        if self.uav_number == 4:
            raise NotImplementedError()
        
        self.uav_coordinate = __uav_coordinate[self.uav_number]


        # 用于生成数据集的设置
        self.board_x_size = 600
        self.board_y_size = 800

        
    def generate_dataset(self, n=5000, saving_path=None):
        la = [] #训练数据

        # features:
        for _ in tqdm(range(n)):
            lm = []

            # 生成用户坐标：(x, y, z)
            for _ in range(self.user_number):
                lm.append(np.random.uniform(0, self.board_x_size))
                lm.append(np.random.uniform(0, self.board_y_size))
                lm.append(0)

            # 生成task info   
            for _ in range(self.user_number):
                data = round(np.random.uniform(3e2, 5e2)) #用户平均任务大小 Mbit/slot单位时间多少Mbit
                lm.append(data)

                cpb = np.random.randint(40,150)*1e-3 # 单位比特计算次数为 40-150 cycles
                data = np.round(data * cpb)#任务需要的cpu数量
                lm.append(data)

                lm.append(np.random.randint(np.ceil(data*1e6/125000),np.ceil(data*1e6/125000)+1000))#任务可以接受的延迟 slot

            la.append(lm)

        self.__save_to_csv(la, saving_path)
        
    
    def __save_to_csv(self, data, file_path):
        with open(file_path, mode='w+', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(data)

    def load_config_from_file(self, file_path):
        """
        作用：从配置文件file_path中读取配置
        """

        print('---> loading config from file: "{}"'.format(file_path))

        with open(file_path, 'r') as f:
            for k, v in json.load(f).items():
                # 将所有list转换为ndarray类型
                if type(v) == list:
                    self.__dict__[k] = np.array(v)
                    continue

                self.__dict__[k] = v

        print('Done.')
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

if __name__ == '__main__':
    # 创建对象
    dataObj = DataConfig(load_config_from_path='config.json')
    # dataObj = DataConfig(n_of_user=10, n_of_uav=10)

    # 保存config
    # dataObj.save_config('config.json')
    
    # 生成数据集:
    dataObj.generate_dataset(n=1000, saving_path='./h.csv')