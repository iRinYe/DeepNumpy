# DeepNumpy
## 简介
* 用Numpy实现PyTorch网络的前向代码.
* 通过现成框架训练模型后, 可通过DeepNumpy提取模型参数并转化为**平台无关**, **框架无关**, **系统无关**的网络结构进行测试. 
* 网络预测性能与原生框架一致(参考PyTorch 0.4.1).

## 需求
* ~~系统~~
* ~~框架~~
* ~~平台~~
* ~~GPU~~
* ~~环境~~
* ~~第三方库~~
* **支持Numpy库即可**

## 目前支持网络结构

* 线性层 Linear
* 二维卷积 Conv2d
* 长短期记忆网络 LSTM
    * todo stacked LSTM还未完成, 请勿尝试layers_num大于1的LSTM网络
    * todo BiLSTM性能与PyTorch稍微有点不一致
* 门控循环单元网络 GRU
    * todo stacked GRU还未完成, 请勿尝试layers_num大于1的GRU网络
    * todo BiGRU还未完成, 请勿尝试BiGRU网络
    
* todo GPU加速

注: 示例代码中的模型结构不是为了预测, 只是为了对比PyTorch与DeepNumpy的效果.

**仅供实验, 切勿实用**

For experiment only, **DO NOT APPLY IT INTO ANY REAL PROJECT**.

<p align="right">Coded By iRinYe.CN</p>