# @Time    : 2020/5/22 11:36
# @Author  : iRinYe
# @Email   : YeYilinCN@outlook.com
# @File    : CNN
# @Software: PyCharm

"""
    利用Numpy实现CNN
"""
import time

import torch
import numpy as np
from sklearn.datasets import load_digits

import DeepNumpy
from lib import getDataLoader, train, test, score


class Model(torch.nn.Module):
    # 模型定义
    def __init__(self):
        super(Model, self).__init__()
        self.CNN = torch.nn.Conv2d(in_channels=3, out_channels=10, kernel_size=(4, 2), padding=(3, 1), stride=(2, 2))
        self.FC = torch.nn.Linear(in_features=300, out_features=10)

    def forward(self, x):
        batch_size = x.shape[0]
        temp = self.CNN(x)
        temp = self.FC(temp.view(batch_size, -1))
        return torch.sigmoid(temp)


if __name__ == "__main__":
    # 主函数入口
    x, y = load_digits(return_X_y=True)

    # 测试多Channel的情况
    x = np.hstack((x, x, x))

    train_len = int(len(x) / 10 * 7)
    test_len = len(x) - train_len

    dl = getDataLoader(x[:train_len].reshape(-1, 3, 8, 8), y[:train_len], 30, True)
    model = Model()

    model = train(model=model, dataloader=dl, EPOCH=10, loss="CEP")

    # PyTorch Test
    start = time.perf_counter()
    dl = getDataLoader(x[train_len:].reshape(-1, 3, 8, 8), y[train_len:], test_len, False)
    result1 = test(model, dl)
    speed1 = time.perf_counter() - start

    # Numpy Test
    start = time.perf_counter()
    weight_dict = DeepNumpy.getModelWeight(model)

    temp = DeepNumpy.Conv2d(x[train_len:].reshape(-1, 3, 8, 8), weight_dict['CNN.weight'], weight_dict['CNN.bias'], padding=(3, 1), stride=(2, 2))
    temp = DeepNumpy.Flatten(temp)
    temp = DeepNumpy.Linear(temp, weight_dict['FC.weight'], weight_dict['FC.bias'])
    temp = DeepNumpy.Sigmoid(temp)

    result2 = temp[:, 1].reshape(-1, 1)
    speed2 = time.perf_counter() - start

    score(y[train_len:], result1, result2, speed1, speed2)
