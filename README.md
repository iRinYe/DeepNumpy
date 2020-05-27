# DeepNumpy
用Numpy实现PyTorch网络的前向代码, 通过现成框架训练模型后, 可通过DeepNumpy提取模型参数并转化为平台无关, 框架无关的网络结构进行测试. 网络预测性能与原生框架一致(参考PyTorch 0.4.1).

需求:
* 支持Numpy库即可

目前支持:

* 线性层 Linear
* 二维卷积 Conv2d
* 长短期记忆网络 LSTM
* 门控循环单元网络 GRU

注: 示例代码中的模型结构不是为了预测, 只是为了对比PyTorch与DeepNumpy的效果.

Coded By iRinYe.CN

**仅供实验, 切勿实用**

For experiment only, **DO NOT APPLY IT INTO ANY REAL PROJECT**.