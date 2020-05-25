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


def Conv2d(x, CNN_filter, CNN_bias, stride=(1, 1), padding=(0, 0)):
    """
    利用Numpy实现CNN
    PyTorch-GPU AUC: 0.95182724, Time used:0.00299s
    DeepNumpy-CPU AUC: 0.95182724, Time used:0.00157s
    speed1:speed2 = 1.898

    :param x: 输入矩阵[batch, channel, H, W]
    :param CNN_filter: 卷积滤片的权重
    :param CNN_bias: 卷积滤片的偏置
    :param stride: 步长, 默认为(1, 1)
    :param padding: 填充, 默认为(1, 1)
    :return: 卷积后的结果数组[batch, filter, channel, H, W]
    """
    filter_num, channel, kernel_h, kernel_w = CNN_filter.shape
    batch_size, input_h, input_w = x.shape
    padding_h, padding_w = padding

    x = np.pad(x, ((0, 0), (padding_h, padding_h), (padding_w, padding_w)), 'constant', constant_values=0)
    Fh = int((input_h - kernel_h + 2 * padding_h) / stride[0] + 1)
    Fw = int((input_w - kernel_w + 2 * padding_w) / stride[1] + 1)

    x = x.reshape(batch_size, 1, channel, input_h + 2 * padding_h, input_w + 2 * padding_w)
    b_result = None

    feature_map = None
    for i in range(Fh):  # row start index
        row_temp = None
        for j in range(Fw):  # col start index

            current_field = x[:, :, :, i: i + kernel_h, j: j + kernel_w]              # [batch, 1, channel, kernel_size[0], kernel_size[1]]
            temp = np.multiply(current_field, CNN_filter)                             # [batch, 1, channel, kernel_size[0], kernel_size[1]]
            temp = np.sum(temp, axis=(-2, -1))                                        # [batch, filter num, channel]
            temp = temp + CNN_bias.reshape(1, filter_num, channel)                    # [batch, filter num, channel]
            temp = temp.reshape(batch_size, filter_num, channel, 1, 1)

            row_temp = np.concatenate((row_temp, temp), axis=-1) if row_temp is not None else temp
        feature_map = np.concatenate((feature_map, row_temp), axis=-2) if feature_map is not None else row_temp
    b_result = np.concatenate((b_result, feature_map), axis=0) if b_result is not None else feature_map     # [batch, filter num, channel, Fh, Fw]

    return np.sum(b_result, axis=2)     # [batch, filter num, Fh, Fw] todo 不同channel之间是加法?

