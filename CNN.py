# @Time    : 2020/5/22 11:36
# @Author  : iRinYe
# @Email   : YeYilinCN@outlook.com
# @File    : FC
# @Software: PyCharm

"""
    利用Numpy实现CNN
    PyTorch-GPU AUC: 0.95182724, Time used:0.00299s
    DeepNumpy-CPU AUC: 0.95182724, Time used:0.00157s
    speed1:speed2 = 1.898
"""
import time

import torch
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score

import DeepNumpy
from lib import getDataLoader, train, test, getModelWeight


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
    dl = getDataLoader(x[train_len:].reshape(-1, 1, 5, 6), y[train_len:], 30, False)
    start = time.perf_counter()
    result = test(model, dl)
    auc = roc_auc_score(y[train_len:] == 1, result)
    speed1 = time.perf_counter() - start
    print()
    print()
    print()
    print("PyTorch AUC: {}, Time used:{}s".format(round(auc, 8), round(speed1, 5)))

    # Numpy Test
    start = time.perf_counter()
    weight_dict = getModelWeight(model)

    temp = DeepNumpy.Conv2d(x[train_len:].reshape(-1, 5, 6), weight_dict['CNN.weight'], weight_dict['CNN.bias'])
    temp = DeepNumpy.Flatten(temp)
    temp = DeepNumpy.Linear(temp, weight_dict['FC.weight'], weight_dict['FC.bias'])
    temp = DeepNumpy.Sigmoid(temp)

    result = temp[:, 1].reshape(-1, 1)
    auc = roc_auc_score(y[train_len:] == 1, result)
    speed2 = time.perf_counter() - start
    print("DeepNumpy AUC: {}, Time used:{}s".format(round(auc, 8), round(speed2, 5)))

    print("speed1:speed2 = {}".format(round(speed1 / speed2, 3)))
