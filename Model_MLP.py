import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

class MLP(nn.Module):
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
        memory_size=1000
    ):
        super().__init__()
        print('[Model] Model Type: MLP')

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

        # store training cost, 用于画图
        self.cost_his = []

        # 准备训练数据
        # ---------
        self.Memory_size    = memory_size
        self.Memory_counter = 0
        self.data_X = torch.zeros((self.Memory_size, self.input_feature_size), dtype=torch.float32)
        self.data_Y = torch.zeros((self.Memory_size, self.data_config.user_number), dtype=torch.float32)

        self._build_net()
        self._build_opt_tools()

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
    
    def _build_opt_tools(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas = (0.09,0.999), weight_decay=0.0001) 
        self.criterion = nn.NLLLoss()
    
    def save_data_to_memory(self, feature, y):
        """
        将新数据放入Memory,
        如果有空间, 则en-queue
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
        prob, ans_idx = self.predict_answer_index(torch.Tensor(input).float())

        answer = self.convert_answer_index_to_zero_one_answer_vector(ans_idx, data_config)
        
        return prob, answer
    
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

        return predict_prob, ans

    def plot_cost(self, model_name):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his))*self.training_interval, self.cost_his)
        plt.ylabel('Training Loss')
        plt.xlabel('Time Frames')
        plt.savefig('model:{}_train_loss.png'.format(model_name))