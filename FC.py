# @Time    : 2020/5/22 11:36
# @Author  : iRinYe
# @Email   : YeYilinCN@outlook.com
# @File    : FC
# @Software: PyCharm

"""
    利用Numpy实现FC
"""
import time

import torch
from sklearn.datasets import load_digits

import DeepNumpy
from lib import getDataLoader, train, test, score


class Model(torch.nn.Module):
    # 模型定义
    def __init__(self):
        super(Model, self).__init__()
        self.FC = torch.nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        return torch.sigmoid(self.FC(x))


if __name__ == "__main__":
    # 主函数入口
    x, y = load_digits(return_X_y=True)
    train_len = int(len(x) / 10 * 7)
    test_len = len(x) - train_len
    dl = getDataLoader(x[:train_len], y[:train_len], 30, True)
    model = Model()

    model = train(model=model, dataloader=dl, EPOCH=10, loss="CEP")

    # PyTorch Test
    start = time.perf_counter()
    dl = getDataLoader(x[train_len:], y[train_len:], test_len, False)
    result1 = test(model, dl)
    speed1 = time.perf_counter() - start

    # Numpy Test
    start = time.perf_counter()
    weight_dict = DeepNumpy.getModelWeight(model)
    result2 = DeepNumpy.Sigmoid(DeepNumpy.Linear(x[train_len:], weight_dict['FC.weight'], weight_dict['FC.bias']))[:, 1]
    speed2 = time.perf_counter() - start

    score(y[train_len:], result1, result2, speed1, speed2)
