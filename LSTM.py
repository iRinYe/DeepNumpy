# @Time    : 2020/5/22 11:36
# @Author  : iRinYe
# @Email   : YeYilinCN@outlook.com
# @File    : LSTM
# @Software: PyCharm

"""
    利用Numpy实现LSTM
    # PyTorch-GPU AUC: 0.97009967, AUPR: 0.98925309, Time used:0.00267s
    # DeepNumpy-CPU AUC: 0.97009967, AUPR: 0.98925309, Time used:0.00178s
    # DeepNumpy-CPU比PyTorch-GPU快了0.501倍
"""
import time

import torch
from sklearn.datasets import load_breast_cancer

import DeepNumpy
from lib import getDataLoader, train, test, getModelWeight, score


class Model(torch.nn.Module):
    # 模型定义
    def __init__(self):
        super(Model, self).__init__()
        self.LSTM = torch.nn.LSTM(input_size=6, hidden_size=10, num_layers=1, batch_first=True)
        self.FC = torch.nn.Linear(in_features=10, out_features=2)

    def forward(self, x):
        temp = self.LSTM(x)[0][:, -1, :]
        return torch.sigmoid(self.FC(temp))


if __name__ == "__main__":
    # 主函数入口
    x, y = load_breast_cancer(True)
    train_len = int(len(x) / 10 * 9)
    dl = getDataLoader(x[:train_len].reshape(-1, 5, 6), y[:train_len], 30, True)
    model = Model()

    model = train(model=model, dataloader=dl, EPOCH=30, loss="CEP")

    # PyTorch Test
    start = time.perf_counter()
    dl = getDataLoader(x[train_len:].reshape(-1, 5, 6), y[train_len:], 100, False)

    result1 = test(model, dl)
    speed1 = time.perf_counter() - start

    # Numpy Test
    start = time.perf_counter()
    weight_dict = getModelWeight(model)
    temp = DeepNumpy.LSTM(x[train_len:].reshape(-1, 5, 6),
                          weight_dict['LSTM.weight_ih_l0'],
                          weight_dict['LSTM.weight_hh_l0'],
                          weight_dict['LSTM.bias_ih_l0'],
                          weight_dict['LSTM.bias_hh_l0'])

    temp = DeepNumpy.Linear(temp, weight_dict['FC.weight'], weight_dict['FC.bias'])
    result2 = DeepNumpy.Sigmoid(temp)[:, 1]
    speed2 = time.perf_counter() - start

    score(y[train_len:], result1, result2, speed1, speed2)
