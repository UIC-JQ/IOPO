#  #################################################################
#  This file contains the main DROO operations, including building DNN, 
#  Storing data sample, Training DNN, and generating quantized binary offloading decisions.

#  version 1.0 -- February 2020. Written based on Tensorflow 2 by Weijian Pan and 
#  Liang Huang (lianghuang AT zjut.edu.cn)
#  ###################################################################

from __future__ import print_function
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

# DNN network for memory
class MemoryDNN(nn.Module):
    def __init__(
        self,
        input_feature_size,
        output_size,
        hidden_feature_size=256,
        learning_rate = 0.01,
        training_interval=10,
        batch_size=100,
        dropout=0.2,
        data=None,
        data_eng_cost=None,
        split_len=None,
        convert_output_size=None,
        data_config=None
    ):
        super(MemoryDNN, self).__init__()
        self.input_feature_size = input_feature_size
        self.hidden_feature_size = hidden_feature_size
        self.output_size = output_size
        self.convert_output_size = convert_output_size
        self.training_interval = training_interval      # learn every training_interval
        self.lr = learning_rate
        self.batch_size = batch_size
        self.dropout_rate = dropout
        self.split_len = split_len
        self.data_config = data_config

        # store all binary actions
        self.enumerate_actions = []

        # store training cost, 用于画图
        self.cost_his = []

        # 准备训练数据
        data = torch.Tensor(data)
        self.data_X = data[:, 0: self.input_feature_size]
        self.data_Y = data[:, self.input_feature_size:]
        self.NUM_OF_DATA = len(data)
        self.data_ENG_COST = torch.Tensor(data_eng_cost)

        # construct memory network
        self._build_net()
        self._build_opt_tools()

    def _build_net(self):
        # (input_size, h//2) -> (h//2, h) -> (h, h) * 5 -> (h, h//2) -> (h//2, output_size)
        self.model = nn.Sequential(
                nn.Linear(self.input_feature_size, self.hidden_feature_size // 2),
                nn.Tanh(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_feature_size // 2, self.hidden_feature_size),
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
                nn.Linear(self.hidden_feature_size, self.hidden_feature_size //2),
                nn.Tanh(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_feature_size // 2, self.output_size),
        )

    
    def _build_opt_tools(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr,betas = (0.09,0.999),weight_decay=0.0001) 
        self.criterion = nn.NLLLoss()
    
    def train(self, flag_regenerate_better_sol=False, eng_cost_func=None):
        # 随机选择batch_size行数据
        sample_index = np.random.choice(self.NUM_OF_DATA, size=self.batch_size)

        # 获取相应的x, y
        x = self.data_X[sample_index]
        y = self.data_Y[sample_index]


        predict = self.model(x)
        # 将predicth(batch_size, uav_num + 1)
        predict = torch.reshape(predict, (-1, self.split_len))

        # 对这些训练数据生成更好的reference answer
        if flag_regenerate_better_sol:
            print('-> Re-Generating Better Solutions')
            # 将数据复原
            split_by_data_pair = torch.tensor_split(predict, predict.shape[0] // self.data_config.user_number, dim=0)
            assert split_by_data_pair[0].shape == (self.data_config.user_number, self.split_len)
            self.better_sol_count = 0

            for idx, each_data_pair_predict in enumerate(split_by_data_pair):
                # 愿数据的位置
                ori_idx = sample_index[idx]
                # 计算predict probability
                probs = nn.functional.softmax(each_data_pair_predict, dim=1)
                # 生成更好的y
                better_sol, eng_cost = self.regenerate_better_solution(probs, self.data_ENG_COST[ori_idx], eng_cost_func, ori_idx)

                if better_sol is not None:
                    # 替换data pair
                    self.data_Y[ori_idx] = better_sol
                    self.data_ENG_COST[ori_idx] = eng_cost
                    # 替换现在的训练数据
                    y[idx] = better_sol
            
            print('Generate {} better solutions, ratio: {:.2f}'.format(self.better_sol_count, (100 * self.better_sol_count) / self.batch_size))

        # 计算LOSS:
        predict = nn.functional.log_softmax(predict, dim=1)
        # 将(batch_size, n_of_user) 垂直展开，结果为b_size * n_of_user行，1列.
        y = torch.reshape(y, (-1, ))
        loss = self.criterion(predict, y.long())

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
        x = torch.Tensor(input)

        predict = self.model(x)
        answer = np.zeros(self.convert_output_size)

        predict = torch.reshape(predict, (-1, self.split_len))
        predict = nn.functional.softmax(predict, dim=1)
        ans = torch.argmax(predict, dim=1)

        for i, idx in enumerate(ans):
            base = i * data_config.uav_number
            if idx == 0:
                continue
            answer[idx + base - 1] = 1
        
        return answer


    def decode(self, h, k = 1, mode = 'OP'):
        # to have batch dimension when feed into Tensor
        h = torch.Tensor(h[np.newaxis, :])
        self.model.eval()
        m_pred = self.model(h)
        m_pred = m_pred.detach().numpy()
        if mode == 'OP':
            return self.knm(m_pred[0], k) #op
        elif mode == 'KNN':
            return self.knn(m_pred[0], k) #knn
        elif mode == 'MP':
            return self.mp(m_pred[0], k) #MP
        else:
            print("The action selection must be 'OP' or 'KNN'")

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
        # print(threshold_rank)
        # print(idx_list)

        new_sol = torch.zeros(self.convert_output_size)
        new_sol_idx = torch.argmax(predict_prob, dim=1)
        T = []
        for row, idx in enumerate(new_sol_idx):
            if idx == 0:
                T.append(0)
                continue

            if predict_prob[row][idx] >= threshold_v:
                T.append(idx)
                new_idx = row * (self.split_len - 1) + (idx - 1)
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
        plt.show()

