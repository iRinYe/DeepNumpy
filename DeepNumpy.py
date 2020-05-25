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


def Flatten(x):
    return x.reshape(x.shape[0], -1)


def Linear(x, weight, bias):
    return np.dot(x, weight.T) + bias


def Conv2d(x, CNN_filter, CNN_bias, stride=(1, 1), padding=0):
    """
    二维卷积
    :param x: 输入矩阵[batch, channel, H, W]
    :param CNN_filter: 卷积滤片的权重
    :param CNN_bias: 卷积滤片的偏置
    :param stride: 步长, 默认为(1, 1)
    :param padding: 填充, 默认为(1, 1), 该参数暂时闲置
    :return: 卷积后的结果数组[batch, filter, channel, H, W]
    """
    kernel_size = (CNN_filter.shape[2], CNN_filter.shape[3])
    channel = CNN_filter.shape[1]

    input_shape = x.shape
    FM_size = (input_shape[1], input_shape[2])

    Fh = int((FM_size[0] - kernel_size[0] + 2 * padding) / stride[0] + 1)
    Fw = int((FM_size[1] - kernel_size[1] + 2 * padding) / stride[1] + 1)

    x = x.reshape(input_shape[0], channel, input_shape[1], input_shape[2])

    b_result = None

    for b in range(input_shape[0]):
        # 一批中的第b个样本
        # todo 碰到多个channel的情况还没测试 感觉Channel维度的运算可能有问题
        feature_map = None
        for i in range(Fh):  # row start index
            row_temp = None
            for j in range(Fw):  # col start index

                current_field = x[b, :, i: i + kernel_size[0], j: j + kernel_size[1]]  # [channel, h, w]
                temp = np.multiply(current_field, CNN_filter)
                temp = np.sum(temp, axis=(2, 3))  # [batch, filter num, channel, 1, 1]
                temp = temp + CNN_bias.reshape(temp.shape)  # [batch, filter num, channel, 1, 1]
                temp = temp.reshape(1, CNN_filter.shape[0], CNN_filter.shape[1], 1, 1)

                row_temp = np.concatenate((row_temp, temp), axis=-1) if row_temp is not None else temp
            feature_map = np.concatenate((feature_map, row_temp), axis=-2) if feature_map is not None else row_temp
        b_result = np.concatenate((b_result, feature_map), axis=0) if b_result is not None else feature_map

    return b_result
