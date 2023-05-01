import torch
from memoryPyTorch import MemoryDNN

if __name__ == '__main__':
    print(111)
    path1 = "/Users/Helen/Documents/Mphi/code/UAV-IRS V2/result/"
    model_path = path1 + 'model_param.pt'

    M = 6
    U = 3
    K = N = U * 6
    out = M * U
    Memory = 1024

    model = MemoryDNN(net = [K, 120, 80, out],
                    memory_size=Memory
                    )
    
    model.load_state_dict(torch.load(model_path))

    data_X, data_Y = test_data()

    predict = model.decode(data_X)
    
    score = eval_metrics(predict, data_Y)

    # for var_name in model.state_dict():
        # print(var_name, "\t", model.state_dict()[var_name])

    # model = torch.load(model_path)
    # print(model.)