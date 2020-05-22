# @Time    : 2020/5/22 14:52
# @Author  : iRinYe
# @Email   : YeYilinCN@outlook.com
# @File    : DeepNumpy
# @Software: PyCharm

"""
    DeepNumpy的网络
"""

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def Linear(x, weight):
    return np.dot(x, weight['FC.weight'].T) + weight['FC.bias']