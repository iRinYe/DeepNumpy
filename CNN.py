# @Time    : 2020/5/22 11:36
# @Author  : iRinYe
# @Email   : YeYilinCN@outlook.com
# @File    : FC
# @Software: PyCharm

"""
    利用Numpy实现CNN
    # PyTorch-GPU AUC: 0.94352159, AUPR: 0.97740254, Time used:0.00205s
    # DeepNumpy-CPU AUC: 0.94352159, AUPR: 0.97740254, Time used:0.00103s
    # DeepNumpy-CPU比PyTorch-GPU快了0.989倍
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
        self.CNN = torch.nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3)
        self.FC = torch.nn.Linear(in_features=5 * 3 * 4, out_features=2)

    def forward(self, x):
        temp = self.CNN(x)
        temp = self.FC(temp.view(-1, 5 * 3 * 4))
        return torch.sigmoid(temp)


if __name__ == "__main__":
    # 主函数入口
    x, y = load_breast_cancer(True)
    train_len = int(len(x) / 10 * 9)
    dl = getDataLoader(x[:train_len].reshape(-1, 1, 5, 6), y[:train_len], 30, True)
    model = Model()

    model = train(model=model, dataloader=dl, EPOCH=30, loss="CEP")

    # PyTorch Test
    start = time.perf_counter()
    dl = getDataLoader(x[train_len:].reshape(-1, 1, 5, 6), y[train_len:], 30, False)
    result1 = test(model, dl)
    speed1 = time.perf_counter() - start

    # Numpy Test
    start = time.perf_counter()
    weight_dict = getModelWeight(model)

    temp = DeepNumpy.Conv2d(x[train_len:].reshape(-1, 5, 6), weight_dict['CNN.weight'], weight_dict['CNN.bias'])
    temp = DeepNumpy.Flatten(temp)
    temp = DeepNumpy.Linear(temp, weight_dict['FC.weight'], weight_dict['FC.bias'])
    temp = DeepNumpy.Sigmoid(temp)

    result2 = temp[:, 1].reshape(-1, 1)
    speed2 = time.perf_counter() - start

    score(y[train_len:], result1, result2, speed1, speed2)
