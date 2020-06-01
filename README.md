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
    
## 实验室
### DeepNumpy-GPU
> cuda加速目前可能不稳定, 请斟酌使用

**DeepNumpy目前支持实验性的Cuda加速**

如有此需求请切换至"cuda加速支持"分支

#### 目前支持网络结构

* 线性层 Linear
* 二维卷积 Conv2d
* 长短期记忆网络 LSTM
    * todo 暂不支持, 请勿在DeepNumpy-GPU中使用LSTM网络
* 门控循环单元网络 GRU
    * todo 暂不支持, 请勿在DeepNumpy-GPU中使用GRU网络

#### 需求
* 在使用DeepNumpy-GPU前, 请确保配置了cuda环境, 并安装了对应的cupy包.
以cuda92为例:
    ```
    pip install cupy-cuda92
    ```


注: 示例代码中的模型结构不是为了预测, 只是为了对比PyTorch与DeepNumpy的效果.

---
**仅供实验, 切勿实用**

For experiment only, **DO NOT APPLY IT INTO ANY REAL PROJECT**.

<p align="right">Coded By iRinYe.CN</p>