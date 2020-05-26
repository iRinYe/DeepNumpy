# @Time    : 2020/5/22 14:52
# @Author  : iRinYe
# @Email   : YeYilinCN@outlook.com
# @File    : DeepNumpy
# @Software: PyCharm

"""
    DeepNumpy的网络
"""

import numpy as np


def getModelWeight(model):
    """
    将模型的参数转化成数组dict, key名为网络中每层的名字
    :param model: PyTorch模型文件
    :return: weight_dict 转换好的权重dict
    """
    weight_dict = dict(model.state_dict())

    for key in dict(weight_dict):
        weight_dict[key] = weight_dict[key].cpu().numpy()

    return weight_dict


def Sigmoid(x):
    np.seterr(divide='ignore', invalid='ignore', over='ignore')
    # todo RuntimeWarning: overflow encountered in exp 暂时没定位到问题, 先把溢出警告屏蔽了
    return 1 / (1 + np.exp(-x))


def ReLU(x):
    return np.maximum(0, x)


def Tanh(x):
    np.seterr(divide='ignore', invalid='ignore', over='ignore')
    a = np.exp(x) - np.exp(-x)
    b = np.exp(x) + np.exp(-x)

    b[np.isinf(b)] = np.inf
    a[np.isinf(a)] = np.inf

    temp = a / b

    temp[np.isposinf(a)] = 1
    temp[np.isneginf(a)] = -1

    if np.sum(np.isnan(temp)) > 0:
        print("Tanh中的元素出现空值")

    return temp


def Flatten(x):
    return x.reshape(x.shape[0], -1)


def Linear(x, weight, bias):
    """
    利用Numpy实现FC

    :param x: 输入向量
    :param weight: 权重矩阵
    :param bias: 偏置向量
    :return: 结果向量
    """
    return np.dot(x, weight.T) + bias


def Conv2d(x, CNN_filter, CNN_bias, stride=(1, 1), padding=(0, 0)):
    """
    利用Numpy实现CNN

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


def LSTM(x, LSTM_weight_i, LSTM_weight_h, LSTM_bias_i, LSTM_bias_h):
    """
    利用Numpy实现LSTM

    :param x: 输入
    :param LSTM_weight_i: 输入状态的权重
    :param LSTM_weight_h: 隐藏状态的权重
    :param LSTM_bias_i: 输入状态的偏置
    :param LSTM_bias_h: 隐藏状态的偏置
    :return: 最后一个时刻的输出
    """
    batch_size, T, input_size = x.shape
    hidden_size = int(LSTM_bias_i.size / 4)

    h_t = np.zeros((batch_size, hidden_size))
    c_t = h_t / 1

    for t in range(T):
        temp_t = np.dot(x[:, t, :], LSTM_weight_i.T) + LSTM_bias_i + np.dot(h_t, LSTM_weight_h.T) + LSTM_bias_h
        temp_t = temp_t.reshape(batch_size, 4, -1)
        i_t, f_t, g_t, o_t = (Sigmoid(temp_t[:, i, :]) if i != 2 else Tanh(temp_t[:, 2, :]) for i in range(4))
        c_t = f_t * c_t + i_t * g_t
        h_t = o_t * Tanh(c_t)

    return h_t
