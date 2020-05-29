# @Time    : 2020/5/22 14:52
# @Author  : iRinYe
# @Email   : YeYilinCN@outlook.com
# @File    : DeepNumpy
# @Software: PyCharm

"""
    DeepNumpy的网络
"""

import cupy as np
# import numpy


def getModelWeight(model):
    """
    将模型的参数转化成数组dict, key名为网络中每层的名字
    :param model: PyTorch模型文件
    :return: weight_dict 转换好的权重dict
    """

    try:
        # 传了模型参数
        weight_dict = dict(model)
    except TypeError:
        try:
            weight_dict = dict(model.state_dict())
        except AttributeError:
            print("请检查模型参数，目前允许传入PyTorch的模型与模型参数OrderedDict")
            print("传入的是:{}".format(type(model)))
            exit()

    for key in weight_dict:
        weight_dict[key] = weight_dict[key].cpu().numpy()

    return weight_dict


def Sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ReLU(x):
    return np.maximum(0, x)


def Tanh(x):
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


def Linear(x, weight, bias=None):
    """
    利用Numpy实现FC

    :param x: 输入向量
    :param weight: 权重矩阵
    :param bias: 偏置向量
    :return: 结果向量
    """

    x, weight, bias = GPU(x, weight, bias)

    if bias is not None:
        return np.dot(x, weight.T) + bias
    else:
        return np.dot(x, weight.T)


def Conv2d(x, CNN_filter, CNN_bias=None, stride=(1, 1), padding=(0, 0)):
    """
    利用Numpy实现CNN

    :param x: 输入矩阵[batch, channel, H, W]
    :param CNN_filter: 卷积滤片的权重
    :param CNN_bias: 卷积滤片的偏置
    :param stride: 步长, 默认为(1, 1)
    :param padding: 填充, 默认为(1, 1)
    :return: 卷积后的结果数组[batch, filter, channel, H, W]
    """
    filter_num, filter_channel, kernel_h, kernel_w = CNN_filter.shape
    batch_size, input_channel, input_h, input_w = x.shape

    assert filter_channel == input_channel, "filter的channel应与input的channel相同！"

    padding_h, padding_w = padding

    x = np.pad(x, ((0, 0), (0, 0), (padding_h, padding_h), (padding_w, padding_w)), 'constant', constant_values=np.asarray(0))
    Fh = int((input_h - kernel_h + 2 * padding_h) / stride[0] + 1)
    Fw = int((input_w - kernel_w + 2 * padding_w) / stride[1] + 1)

    x = x.reshape(batch_size, 1, input_channel, input_h + 2 * padding_h, input_w + 2 * padding_w)

    feature_map = None
    for i in range(Fh):  # row start index
        row_temp = None
        row_i = i * stride[0]
        for j in range(Fw):  # col start index
            col_i = j * stride[1]
            current_field = x[:, :, :, row_i: row_i + kernel_h,
                            col_i: col_i + kernel_w]  # [batch, 1, i_channel, kernel_size[0], kernel_size[1]]
            temp = np.multiply(current_field,
                               CNN_filter)  # [batch, filter num, i_channel, kernel_size[0], kernel_size[1]]
            temp = np.sum(temp, axis=(2, 3, 4))  # [batch, filter num]
            if CNN_bias is not None:
                temp = temp + CNN_bias.reshape(1, filter_num)  # [batch, filter num]
            temp = temp.reshape(batch_size, filter_num, 1, 1)

            row_temp = np.concatenate((row_temp, temp), axis=-1) if row_temp is not None else temp
        feature_map = np.concatenate((feature_map, row_temp), axis=-2) if feature_map is not None else row_temp

    return feature_map  # [batch, filter num, Fh, Fw]


def LSTMcell(input, batch_size, hidden_size, Wi, Wh, bi, bh, T):
    input, batch_size, hidden_size, Wi, Wh, bi, bh, T = GPU(input, batch_size, hidden_size, Wi, Wh, bi, bh, T)

    h_t = np.zeros((batch_size, hidden_size))
    c_t = h_t / 1

    for t in range(T):

        temp_t = Linear(input[:, t, :], Wi, bi) + Linear(h_t, Wh, bh)
        temp_t = temp_t.reshape(batch_size, 4, -1)
        i_t, f_t, g_t, o_t = (Sigmoid(temp_t[:, i, :]) if i != 2 else Tanh(temp_t[:, i, :]) for i in range(4))
        c_t = f_t * c_t + i_t * g_t
        h_t = o_t * Tanh(c_t)

    return h_t, c_t


def LSTM(x, weight_i, weight_h, bias_i, bias_h, bidirectional=False):
    """
    利用Numpy实现LSTM

    :param x: 输入
    :param weight_i: 输入状态的权重
    :param weight_h: 隐藏状态的权重
    :param bias_i: 输入状态的偏置
    :param bias_h: 隐藏状态的偏置
    :param bidirectional: 是否双向
    :return: 最后一个时刻的输出
    """

    if isinstance(weight_i, tuple):
        if bidirectional is False:
            assert len(weight_i) == len(weight_h) == len(bias_i) == len(bias_h), "请确保LSTM参数的层数是一致的"
            layer_num = int(len(weight_i))
            hidden_size = int(bias_i[0].size / 4)
            # todo stacked LSTM
        else:
            layer_num = int(len(weight_i) / 2)
            assert len(weight_i) == len(weight_h) == len(bias_i) == len(bias_h) == layer_num * 2, "请确保正确传递双向LSTM的参数"
            hidden_size = int(bias_i[0].size / 4 / layer_num)
    else:
        layer_num = 1
        hidden_size = int(bias_i.size / 4)

    batch_size, T, input_size = x.shape

    for layer in range(layer_num):
        if bidirectional is True:
            h_t = None
            c_t = None
            for ss in range(2):
                index = 2 * layer + ss
                temp_h, temp_c = LSTMcell(x, batch_size, hidden_size,
                                          weight_i[index], weight_h[index], bias_i[index], bias_h[index], T)

                h_t = np.concatenate((h_t, temp_h), axis=1) if h_t is not None else temp_h
                c_t = np.concatenate((c_t, temp_c), axis=1) if c_t is not None else temp_c

        else:
            h_t, c_t = LSTMcell(x, batch_size, hidden_size,
                                weight_i, weight_h, bias_i, bias_h, T)

    return h_t


def GRU(x, weight_i, weight_h, bias_i, bias_h):
    """
    利用Numpy实现GRU

    :param x: 输入
    :param weight_i: 输入状态的权重
    :param weight_h: 隐藏状态的权重
    :param bias_i: 输入状态的偏置
    :param bias_h: 隐藏状态的偏置
    :return: 最后一个时刻的输出
    """
    batch_size, T, input_size = x.shape
    hidden_size = int(bias_i.size / 3)

    h_t = np.zeros((batch_size, hidden_size))

    for t in range(T):

        temp_input = Linear(x[:, t, :], weight_i, bias_i)
        temp_hidden = Linear(h_t, weight_h, bias_h)

        temp_input = temp_input.reshape(batch_size, 3, -1)
        temp_hidden = temp_hidden.reshape(batch_size, 3, -1)

        r_t, z_t = (Sigmoid(temp_input[:, i, :] + temp_hidden[:, i, :]) for i in range(2))
        n_t = Tanh(temp_input[:, 2, :] + r_t * temp_hidden[:, 2, :])

        h_t = (1 - z_t) * n_t + z_t * h_t

    return h_t


def GPU(*args):
    return (cp.asarray(arg) if type(arg) == "" else arg for arg in args)


def CPU(x):
    return cp.asnumpy(x)