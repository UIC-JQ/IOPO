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
import sys

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
        data=None
    ):
        super(MemoryDNN, self).__init__()
        self.input_feature_size = input_feature_size
        self.hidden_feature_size = hidden_feature_size
        self.output_size = output_size

        self.training_interval = training_interval      # learn every training_interval
        self.lr = learning_rate
        self.batch_size = batch_size
        self.dropout_rate = dropout

        # store all binary actions
        self.enumerate_actions = []

        # stored # memory entry
        self.step_count = 1

        # store training cost
        self.cost_his = []

        # initialize zero memory [h, m]
        # self.memory = np.zeros((self.memory_size, self.input_feature_size + self.output_size))
        self.data = torch.Tensor(data)

        # construct memory network
        self._build_net()
        self._build_opt_tools()

    def _build_net(self):
        # (input_size, h//2) -> (h//2, h) -> (h, h) -> (h, h//2) -> (h//2, output_size)
        # self.model = nn.Sequential(
        #         nn.Linear(self.input_feature_size, self.hidden_feature_size // 2),
        #         nn.ReLU(),
        #         nn.Dropout(self.dropout_rate),
        #         nn.Linear(self.hidden_feature_size // 2, self.hidden_feature_size),
        #         nn.ReLU(),
        #         nn.Dropout(self.dropout_rate),
        #         nn.Linear(self.hidden_feature_size, self.hidden_feature_size),
        #         nn.ReLU(),
        #         nn.Dropout(self.dropout_rate),
        #         nn.Linear(self.hidden_feature_size, self.hidden_feature_size //2),
        #         nn.ReLU(),
        #         nn.Dropout(self.dropout_rate),
        #         nn.Linear(self.hidden_feature_size // 2, self.output_size),
        #         nn.Sigmoid()
        # )

        self.model = nn.Sequential(
            nn.Linear(self.input_feature_size, self.hidden_feature_size),
            nn.ReLU(),
            nn.Linear(self.hidden_feature_size, self.hidden_feature_size),
            nn.ReLU(),
            nn.Linear(self.hidden_feature_size, self.output_size),
            nn.ReLU(),
            nn.Sigmoid()
        )
    
    def _build_opt_tools(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr,betas = (0.09,0.999),weight_decay=0.0001) 
        self.criterion = nn.BCELoss()
    
    def train(self):
        if self.step_count % self.training_interval == 0:
            self.learn()
        
        self.step_count += 1

#     def remember(self, h, m):
#         # replace the old memory with new memory
#         idx = self.memory_counter % self.memory_size
#         self.memory[idx, :] = np.hstack((h, m))

#         self.memory_counter += 1

#     def encode(self, h, m):
#         # encoding the entry
#         self.remember(h, m)
#         # train the DNN every 10 step
# #        if self.memory_counter> self.memory_size / 2 and self.memory_counter % self.training_interval == 0:

#         # NOTE: train the network every [self.training_interval] steps.
#         if self.memory_counter % self.training_interval == 0:
#             self.learn()
        

    def learn(self):
        # 随机选择batch_size行数据
        sample_index = np.random.choice(len(self.data), size=self.batch_size)
        # 将batch_size行数据存入batch_memory
        sampled_data_pairs = self.data[sample_index, :]

        # 解析X, Y
        # [0, self.net[0]) 为 input feature
        # [self.net[0]: ] 为 标准答案
        x = sampled_data_pairs[:, 0: self.input_feature_size]
        y = sampled_data_pairs[:, self.input_feature_size:]
        predict = self.model(x)

        # 反向传播:
        # optimizer = optim.Adam(self.model.parameters(), lr=self.lr,betas = (0.09,0.999),weight_decay=0.0001) 
        # criterion = nn.BCELoss()
        # self.model.train()
        # loss = criterion(predict, y)
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        loss = self.criterion(predict, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        cost = loss.item()

        assert(cost > 0)
        self.cost_his.append(cost)
    
    def save_model(self, save_path):
        torch.save(self, save_path)
    
    @staticmethod
    def load_model(model_path):
        return torch.load(model_path)
    
    def generate_answer(self, input, data_config):
        x = torch.Tensor(input)
        threshold = 0.5

        predict = self.model(x)
        print(predict)
        answer = np.zeros(self.output_size)

        for i in range(data_config.user_number):
            idx = i * data_config.uav_number
            prob = predict[idx: idx + data_config.uav_number]
            arg_idx = torch.argmax(prob)
            if predict[arg_idx + idx] > threshold:
                answer[arg_idx + idx] = 1
        
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

