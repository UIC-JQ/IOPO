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
        self.encoder_h = nn.Sequential(
            nn.Linear(self.input_feature_size, self.hidden_feature_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_feature_size, self.hidden_feature_size),
            nn.Sigmoid()
        )

        self.encoder_c = nn.Sequential(
            nn.Linear(self.input_feature_size, self.hidden_feature_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_feature_size, self.hidden_feature_size),
            nn.Sigmoid()
        )

        self.dec_enc = nn.Sequential(
            nn.Linear(self.each_person_feature_size, self.hidden_feature_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_feature_size, self.hidden_feature_size),
            nn.ReLU()
        )

        self.uav_overload_enc = nn.Sequential(
            nn.Linear(self.data_config.uav_number, self.hidden_feature_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_feature_size, self.hidden_feature_size),
            nn.ReLU()
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
        # encoded = self.encoder(x)
        dec_h = self.encoder_h(x)
        dec_c = self.encoder_c(x)

        # 2nd stage: decode:
        # 1. 初始化 decoder hidden states (用encoder的output)
        # dec_h = enc_hidden[0].reshape(self.batch_size, -1)
        # dec_c = enc_hidden[1].reshape(self.batch_size, -1)

        loss = 0
        # 2. generate answer one by one 
        for i in range(self.data_config.user_number):
            # 第i列的x, 以及answer:
            it_x = x_each_person_features[:, i]
            it_y = y[:, i]
            # 处理无人机的overload信息
            x_uav_capacity_features = torch.div(x_uav_capacity_features, overload_factor)

            # 开始输入模型
            dec_h, dec_c = self.decoder(self.dec_enc(it_x), (dec_h, dec_c))
            # output = output.squeeze(0)

            # concatenate在一起
            x_uavs = self.uav_overload_enc(x_uav_capacity_features)
            output = torch.cat([dec_h, x_uavs], dim=1)
            # formula: output = W([output; overload_info]) + b
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
        dec_h = self.encoder_h(x).reshape(-1, self.hidden_feature_size)
        dec_c = self.encoder_c(x).reshape(-1, self.hidden_feature_size)

        # 2nd stage: decode:
        # 初始化 decoder hidden states (用encoder的output)
        # h0 = enc_hidden[0].reshape(1, 1, -1)
        # c0 = enc_hidden[1].reshape(1, 1, -1)
        ans = []

        for i in range(data_config.user_number):
            # 预测第i个人
            it_x = x_each_person_features[:, i]
            x_uav_capacity_features = torch.div(x_uav_capacity_features, overload_factor)

            dec_h, dec_c = self.decoder(self.dec_enc(it_x), (dec_h, dec_c))
            # output, (h0, c0) = self.decoder(it_x.unsqueeze(0), (h0, c0))
            # output = output.squeeze(0)

            x_uavs = self.uav_overload_enc(x_uav_capacity_features)
            output = torch.cat([dec_h, x_uavs], dim=1)
            # formula: output = W([output; overload_info]) + b
            output = self.final_linear_layer(output)

            output_index = torch.argmax(nn.functional.softmax(output, dim=1), dim=1)

            for user_idx, choice_id in enumerate(output_index):
                ans.append(choice_id)
                if choice_id == 0: continue
                overload_factor[user_idx][choice_id - 1] += 1
            
        print([int(i) for i in ans])
        return self.convert_answer_index_to_zero_one_answer_vector(ans, data_config)
    
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

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his))*self.training_interval, self.cost_his)
        plt.ylabel('Training Loss')
        plt.xlabel('Time Frames')
        plt.savefig('train_loss.png')


# DNN network for memory
class MemoryDNN(nn.Module):
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
        which_model='MLP'
    ):
        super().__init__()
        self.model_name = which_model

        # 模型参数
        self.input_feature_size = input_feature_size
        self.hidden_feature_size = hidden_feature_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.dropout_rate = dropout

        # 学习参数:
        self.lr = learning_rate
        self.training_interval = training_interval      # learn every training_interval

        # 生成解参数:
        self.convert_output_size = convert_output_size
        self.choice_len = split_len
        self.data_config = data_config

        # store all binary actions
        self.enumerate_actions = []

        # store training cost, 用于画图
        self.cost_his = []

        # 准备训练数据
        # ---------
        self.Memory_size    = memory_size
        self.Memory_counter = 0
        self.data_X = torch.zeros((self.Memory_size, self.input_feature_size), dtype=torch.float32)
        self.data_Y = torch.zeros((self.Memory_size, self.data_config.user_number), dtype=torch.float32)

        # construct memory network
        if self.model_name == 'MLP':
            print('Model Type: MLP')
            self._build_net()
        elif self.model_name == 'Transformer':
            print('Model Type: Transformer')
            self._build_transformer_net()
        else:
            print('invalid model type')
            exit(0)

        self._build_opt_tools()

        # move everything to gpu/mps
        # self.model.to(torch.device('mps'))
        # self.data_X.to(torch.device('mps'))
        # self.data_Y.to(torch.device('mps'))

    def _build_net(self):
        # (input_feature_size, h) -> (h, h) * 4 -> (h, output_size)
        self.model = nn.Sequential(
                nn.Linear(self.input_feature_size, self.hidden_feature_size),
                nn.Tanh(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_feature_size, self.hidden_feature_size),
                nn.Tanh(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_feature_size, self.hidden_feature_size),
                nn.Tanh(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_feature_size, self.hidden_feature_size),
                nn.Tanh(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_feature_size, self.hidden_feature_size),
                nn.Tanh(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_feature_size, self.hidden_feature_size),
                nn.Tanh(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_feature_size, self.hidden_feature_size),
                nn.Tanh(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_feature_size, self.output_size),
        )
    
    def _build_lstm_net(self):
        self.model = LSTM_Model(input_dim=self.input_feature_size, 
                                hidden_dim=self.hidden_feature_size, output_dim=self.output_size)
    
    def _build_transformer_net(self):
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_feature_size, nhead=4, dim_feedforward=self.hidden_feature_size, dropout=self.dropout_rate)

        self.model = nn.Sequential(
            nn.Linear(self.input_feature_size, self.hidden_feature_size),
            nn.TransformerEncoder(encoder_layer=encoder_layer,
                                  num_layers=4
            ),
            nn.Linear(self.hidden_feature_size, self.output_size)
        )
    
    def _build_opt_tools(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas = (0.09,0.999), weight_decay=0.0001) 
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
        y = self.data_Y[sample_index]

        # shape: (batch_size, output_dim)
        predict = self.model(x)

        # 将predict 转为(N, choice的数量)
        predict = torch.reshape(predict, (-1, self.choice_len))

        # 计算LOSS:
        predict = nn.functional.log_softmax(predict, dim=1)
        # 将(batch_size, y_dim) 垂直展开，shape: (N)
        y = torch.reshape(y, (-1, ))
        loss = self.criterion(predict, y.long()) * self.data_config.user_number
        print(loss)

        # 优化模型参数
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 用于画图
        self.cost_his.append(loss.item())
    
    def save_model(self, save_path):
        torch.save(self, save_path)
    
    @staticmethod
    def load_model(model_path):
        return torch.load(model_path)
    
    def generate_answer(self, input, data_config):
        ans_idx = self.predict_answer_index(torch.Tensor(input).float())

        answer = self.convert_answer_index_to_zero_one_answer_vector(ans_idx, data_config)
        
        return answer
    
    def convert_answer_index_to_zero_one_answer_vector(self, ans_idxs, data_config):
        answer_vector = np.zeros(self.convert_output_size)

        for i, a_idx in enumerate(ans_idxs):
            if a_idx == 0:
                continue

            base = i * data_config.uav_number
            answer_vector[base + a_idx - 1] = 1

        return answer_vector

    
    def predict_answer_index(self, input):
        """
        生成给定input的模型预测answer
        """
        input = torch.Tensor(input).view(1, -1)
        predict = self.model(input)

        predict = torch.reshape(predict, (-1, self.choice_len))
        predict_prob = nn.functional.softmax(predict, dim=1)

        ans = torch.argmax(predict_prob, dim=1).flatten()
        print(ans)

        return ans


    def decode(self, feature, K=1, threshold_v=0.5, eng_compute_func=None, idx=None):
        predict = torch.reshape(self.model(feature), (-1, self.choice_len))
        prob = torch.nn.functional.softmax(predict, dim=1)
        flatten_prob = prob.view(-1, 1)

        threshold_0 = torch.full(flatten_prob.shape, fill_value=threshold_v, dtype=torch.float32)
        threshold_rank, idx_list = torch.sort(nn.functional.pairwise_distance(threshold_0, flatten_prob, p=2))

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

        
        


    # def decode(self, h, k = 1, mode = 'OP'):
    #     # to have batch dimension when feed into Tensor
    #     h = torch.Tensor(h[np.newaxis, :])
    #     self.model.eval()
    #     m_pred = self.model(h)
    #     m_pred = m_pred.detach().numpy()
    #     if mode == 'OP':
    #         return self.knm(m_pred[0], k) #op
    #     elif mode == 'KNN':
    #         return self.knn(m_pred[0], k) #knn
    #     elif mode == 'MP':
    #         return self.mp(m_pred[0], k) #MP
    #     else:
    #         print("The action selection must be 'OP' or 'KNN'")

    def mp(self, m, k = 1):
        """
        model -> predict
        1. sol1: predict -> 最大的和Threshold比，只有大于threshold的被生成为1
        2. 循环K次
            2.1 每次
        """
        threshold = 0.5
        m_list = []
        uav = 6
        user = 3
        #calculate abs 
        m_abs = abs(m-0.5)
        idx_list = np.argsort(m_abs)[:k-1]
        mt = m.reshape(-1, uav)
        max_val_per_row = np.argmax(mt, axis=1)

        b = np.zeros((user, uav), dtype=int)

        for i in range(len(max_val_per_row)):
            if mt[i, max_val_per_row[i]] >= threshold:
                b[i, max_val_per_row[i]] = 1

        b = b.reshape(-1)
        m_list.append(b)

        if k > 1:
            for j in range(k-1):
                b = np.zeros((user, uav), dtype=int)
                for i in range(user):
                    for s in range(uav):
                        if mt[i,s] > m[idx_list[j]]:
                            b[i,s] = 1
                            break
                        elif mt[i,s] == m[idx_list[j]]and m[idx_list[j]] <= 0.5:
                            b[i,s] = 1
                            break
                        #make sure only one 1 in one row
    
                b = b.reshape(-1)
                m_list.append(b)

        # remove duplicated item
        m_set = set(map(tuple, m_list))
        m_new_list = list(map(list, m_set))
        return m_new_list
    
    def regenerate_better_solution(self, predict_prob: torch.Tensor, min_eng_cost: torch.float, eng_cost_function, ori_idx: int):
        # 记录:
        final_ans = None
        reshaped_prob = predict_prob.view(-1, 1)
        threshold_v = float(torch.mean(reshaped_prob))

        # (1, self.output_size)
        threshold_0 = torch.full(reshaped_prob.shape, fill_value=threshold_v, dtype=torch.float32)
        threshold_rank, idx_list = torch.sort(nn.functional.pairwise_distance(threshold_0, reshaped_prob, p=2))

        new_sol = torch.zeros(self.convert_output_size)
        new_sol_idx = torch.argmax(predict_prob, dim=1)
        T = []
        for row, idx in enumerate(new_sol_idx):
            if idx == 0:
                T.append(0)
                continue

            if predict_prob[row][idx] >= threshold_v:
                T.append(idx)
                new_idx = row * (self.choice_len - 1) + (idx - 1)
                new_sol[new_idx] = 1
            else:
                T.append(0)

        _, energy_cost = eng_cost_function(ori_idx, new_sol)
        if min_eng_cost > energy_cost:
            min_eng_cost = energy_cost
            final_ans = torch.tensor(T, dtype=torch.long)
            self.better_sol_count += 1
        
        return final_ans, min_eng_cost
    
    def knm(self, m, k = 1):
        # return k order-preserving binary actions
        m_list = []
        # generate the ﬁrst binary ofﬂoading decision with respect to equation (8)
        m_list.append(1*(m>0.5))

        if k > 1:
            # generate the remaining K-1 binary ofﬂoading decisions with respect to equation (9)
            m_abs = abs(m-0.5)
            idx_list = np.argsort(m_abs)[:k-1]
            for i in range(k-1):
                if m[idx_list[i]] >0.5:
                    # set the \hat{x}_{t,(k-1)} to 0
                    m_list.append(1*(m - m[idx_list[i]] > 0))
                else:
                    # set the \hat{x}_{t,(k-1)} to 1
                    m_list.append(1*(m - m[idx_list[i]] >= 0))
        # print(k)
        # print(m_list)
        return m_list

    def knn(self, m, k = 1):
        # list all 2^N binary offloading actions
        if len(self.enumerate_actions) == 0:
            import itertools
            self.enumerate_actions = np.array(list(map(list, itertools.product([0, 1], repeat=self.net[0]))))

        # the 2-norm
        sqd = ((self.enumerate_actions - m)**2).sum(1)
        idx = np.argsort(sqd)
        return self.enumerate_actions[idx[:k]]


    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his))*self.training_interval, self.cost_his)
        plt.ylabel('Training Loss')
        plt.xlabel('Time Frames')
        plt.savefig('train_loss.png')