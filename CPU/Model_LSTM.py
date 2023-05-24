import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np


class LSTM_Model(nn.Module):
    def __init__(
        self,
        input_feature_size,
        output_size,
        hidden_feature_size=256,
        learning_rate = 0.01,
        training_interval=10,
        batch_size=256,
        dropout=0.2,
        split_len=None,
        convert_output_size=None,
        data_config=None,
        memory_size=1000,
    ):
        super().__init__()
        print('[Model] Model Type: NAIVE LSTM')

        # 生成解参数:
        self.convert_output_size = convert_output_size
        self.choice_len = split_len
        self.data_config = data_config

        # 模型参数
        self.input_feature_size = input_feature_size
        self.hidden_feature_size = hidden_feature_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.dropout_rate = dropout
        self.each_person_feature_size = (input_feature_size - self.data_config.uav_number) // self.data_config.user_number

        # 学习参数:
        self.lr = learning_rate
        self.training_interval = training_interval      # learn every training_interval

        # 准备训练数据
        # ---------
        self.Memory_size    = memory_size
        self.Memory_counter = 0
        self.data_X = torch.zeros((self.Memory_size, self.input_feature_size), dtype=torch.float32)
        self.data_Y = torch.zeros((self.Memory_size, self.data_config.user_number), dtype=torch.float32)

        self._build_model()
        self._build_opt_tools()

        # store training cost, 用于画图
        self.cost_his = []
        
    def _build_model(self):
        self.encoder = nn.Sequential(
            nn.Linear(self.each_person_feature_size, self.hidden_feature_size),
            nn.Tanh(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_feature_size, self.hidden_feature_size),
            nn.Tanh(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_feature_size, self.hidden_feature_size),
            nn.Tanh(),
        )

        self.uav_overload_enc = nn.Sequential(
            nn.Linear(self.data_config.uav_number, self.hidden_feature_size),
            nn.Tanh(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_feature_size, self.hidden_feature_size),
            nn.Tanh(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_feature_size, self.hidden_feature_size),
            nn.Tanh(),
        )

        self.decoder = nn.LSTMCell(input_size=self.hidden_feature_size,
                                   hidden_size=self.hidden_feature_size)

        self.final_linear_layer = nn.Linear(self.hidden_feature_size * 2, self.choice_len)

    def _build_opt_tools(self):
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, betas = (0.09,0.999), weight_decay=0.0001) 
        self.criterion = nn.NLLLoss()
    
    def save_data_to_memory(self, feature, y):
        """
        将新数据放入Memory,
        如果有空间则enque
        如果没有空间，则换出一个memory中的数据
        """
        replace_idx = self.Memory_counter % self.Memory_size
        self.data_X[replace_idx, :] = feature
        self.data_Y[replace_idx, :] = y

        self.Memory_counter += 1

    def encode(self, feature=None, y=None, idx=None):
        self.save_data_to_memory(feature, y)

        if self.Memory_counter > 0 and self.Memory_counter % self.training_interval == 0:
            self.train()
    
    def train(self):
        # 随机选择batch_size行数据
        if self.Memory_counter > self.Memory_size:
            sample_index = np.random.choice(self.Memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.Memory_counter, size=self.batch_size)
        
        # 获取相应的x, y
        x = self.data_X[sample_index]
        x_each_person_features = x[:, : -self.data_config.uav_number]                                                       # shape: (batch_size, feature_dim)
        x_each_person_features = x_each_person_features.reshape(self.batch_size, self.data_config.user_number, -1)          # shape: (batch_size, L, Each_person_feature_dim)
        x_uav_capacity_features = x[:, -self.data_config.uav_number:]                                                       # shape: (batch_size, uav_number)

        y = self.data_Y[sample_index]
        overload_factor = torch.ones((self.batch_size, self.data_config.uav_number), requires_grad=False)

        # 1st stage: encode:
        x_each_person_features = self.encoder(x_each_person_features)
        dec_h = torch.zeros((self.batch_size, self.hidden_feature_size))
        dec_c = torch.zeros((self.batch_size, self.hidden_feature_size))

        # 2nd stage: decode:
        loss = 0
        # 2. generate answer one by one 
        for i in range(self.data_config.user_number):
            # 第i列的x, 以及answer:
            it_x = x_each_person_features[:, i]
            it_y = y[:, i]
            # 处理无人机的overload信息
            f_uav_workload = torch.div(x_uav_capacity_features, overload_factor)
            x_uavs = self.uav_overload_enc(f_uav_workload)

            # 开始输入Decoder
            dec_h, dec_c = self.decoder(it_x, (dec_h, dec_c))

            # concatenate在一起
            output = torch.cat([dec_h, x_uavs], dim=1)
            output = self.final_linear_layer(output)

            # 计算loss
            predict = nn.functional.log_softmax(output, dim=1)
            loss += self.criterion(predict, it_y.long())

            # 更新无人机的overload信息
            for user_idx, uav_id in enumerate(it_y):
                if uav_id == 0: continue
                overload_factor[user_idx][int(uav_id) - 1] += 1
        
        self.cost_his.append(loss.item())
        # 优化模型参数
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def generate_answer(self, x, data_config):
        x = torch.Tensor(x)
        x_each_person_features = x[ : -data_config.uav_number]                                                      # shape: (1, feature_dim)
        x_each_person_features = x_each_person_features.reshape(1, data_config.user_number, -1)                     # shape: (1, L, Each_person_feature_dim)
        x_uav_capacity_features = x[-data_config.uav_number:]                                                       # shape: (1, uav_number)

        overload_factor = torch.ones((1, data_config.uav_number))

        # 1st stage: encode:
        x_each_person_features = self.encoder(x_each_person_features)
        dec_h = torch.zeros(1, self.hidden_feature_size)
        dec_c = torch.zeros(1, self.hidden_feature_size)

        # 2nd stage: decode:
        ans = []
        probs = []
        for i in range(data_config.user_number):
            # 拿到feature:
            it_x = x_each_person_features[:, i]
            f_uav_workload = torch.div(x_uav_capacity_features, overload_factor)
            x_uavs = self.uav_overload_enc(f_uav_workload)

            dec_h, dec_c = self.decoder(it_x, (dec_h, dec_c))

            output = torch.cat([dec_h, x_uavs], dim=1)
            # formula: output = W([output; overload_info]) + b
            output = self.final_linear_layer(output)

            prob   = nn.functional.softmax(output, dim=1)
            probs.append(prob)

            # 计算解的index
            output_index = torch.argmax(prob, dim=1)
            output_index = int(output_index)

            # 更新无人机的overload信息
            ans.append(output_index)
            if output_index != 0:
                overload_factor[0][output_index - 1] += 1

            # output_index = torch.argmax(prob, dim=1)

            # for user_idx, choice_id in enumerate(output_index):
            #     ans.append(choice_id)
            #     if choice_id == 0: continue
            #     overload_factor[user_idx][choice_id - 1] += 1
            
        return torch.vstack(probs), ans, self.convert_answer_index_to_zero_one_answer_vector(ans, data_config)
    
    def convert_answer_index_to_zero_one_answer_vector(self, ans_idxs, data_config):
        answer_vector = np.zeros(self.convert_output_size)

        for i, a_idx in enumerate(ans_idxs):
            if a_idx == 0:
                continue

            base = i * data_config.uav_number
            answer_vector[base + a_idx - 1] = 1

        return answer_vector

    def save_model(self, save_path):
        torch.save(self, save_path)
    
    @staticmethod
    def load_model(model_path):
        return torch.load(model_path)

    def plot_cost(self, save_dir, model_name):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his))*self.training_interval, self.cost_his)
        plt.ylabel('Training Loss')
        plt.xlabel('Time Frames')
        plt.savefig(save_dir + 'model:{}_train_loss.png'.format(model_name))