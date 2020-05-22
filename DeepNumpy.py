# @Time    : 2020/5/22 14:52
# @Author  : iRinYe
# @Email   : YeYilinCN@outlook.com
# @File    : DeepNumpy
# @Software: PyCharm

"""
    DeepNumpy的网络
"""

import numpy as np


def Sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ReLU(x):
    return np.maximum(0, x)


def Tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def Linear(x, weight, bias):
    return np.dot(x, weight.T) + bias


def Conv2d(x, weight):
    # todo 以下参数暂时为超参
    kernel_size = (3, 3)
    stride = (1, 1)
    padding = 0
    channel = 1

    input_shape = x.shape
    FM_size = (input_shape[1], input_shape[2])

    Fh = int((FM_size[0] - kernel_size[0] + 2 * padding) / stride[0] + 1)       # 横向滑动次数
    Fw = int((FM_size[1] - kernel_size[1] + 2 * padding) / stride[1] + 1)       # 纵向滑动次数

    CNN_filter = weight['CNN.weight']
    CNN_bias = weight['CNN.bias']

    x = x.reshape(input_shape[0], channel, input_shape[1], input_shape[2])
    feature_map_List = []
    for b in range(input_shape[0]):
        # 一批中的第b个样本
        for c in range(channel):
            # 第c个channel
            for f in range(CNN_filter.shape[0]):
                # 第f张滤片
                for i in range(Fw):
                    # 横向滑动
                    for j in range(Fh):
                        # 纵向滑动
                        # todo 整不明白了...
                        current_field = x[b, c, i: i+kernel_size[0], j: j+kernel_size[1]]
                        feature_map = np.sum(np.dot(current_field, CNN_filter[f, c, :, :]))
                        feature_map_List.append(feature_map)
        pass