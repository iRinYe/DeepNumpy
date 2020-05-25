# @Time    : 2020/5/22 11:36
# @Author  : iRinYe
# @Email   : YeYilinCN@outlook.com
# @File    : FC
# @Software: PyCharm

"""
    利用Numpy实现FC
    PyTorch-GPU AUC: 0.96346, Time used:0.00318s
    DeepNumpy-CPU AUC: 0.96346, Time used:0.00119s
    speed1:speed2 = 2.664
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
        self.FC = torch.nn.Linear(in_features=30, out_features=15)
        self.FC2 = torch.nn.Linear(in_features=15, out_features=2)

    def forward(self, x):
        return torch.sigmoid(self.FC2(torch.tanh(self.FC(x))))


if __name__ == "__main__":
    # 主函数入口
    x, y = load_breast_cancer(True)
    train_len = int(len(x) / 10 * 9)
    dl = getDataLoader(x[:train_len], y[:train_len], 30, True)
    model = Model()

    model = train(model, dl, 30, "CEP")

    # PyTorch Test
    dl = getDataLoader(x[train_len:], y[train_len:], 30, False)
    start = time.perf_counter()
    result = test(model, dl)
    auc = roc_auc_score(y[train_len:] == 1, result)
    speed1 = time.perf_counter() - start
    print()
    print()
    print()
    print("PyTorch-GPU AUC: {}, Time used:{}s".format(round(auc, 5), round(speed1, 5)))

    # Numpy Test
    start = time.perf_counter()
    weight_dict = getModelWeight(model)
    result = DeepNumpy.Sigmoid(
        DeepNumpy.Linear(
            DeepNumpy.Tanh(
                DeepNumpy.Linear(x[train_len:], weight_dict['FC.weight'], weight_dict['FC.bias'])),
            weight_dict['FC2.weight'], weight_dict['FC2.bias']))[:, 1].reshape(-1, 1)
    auc = roc_auc_score(y[train_len:] == 1, result)
    speed2 = time.perf_counter() - start
    print("DeepNumpy-CPU AUC: {}, Time used:{}s".format(round(auc, 5), round(speed2, 5)))

    print("speed1:speed2 = {}".format(round(speed1 / speed2, 3)))
