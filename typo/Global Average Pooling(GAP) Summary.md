# Global Average Pooling(GAP) Summary

## 1. [Keras Pooling Docs](https://keras.io/layers/pooling/)

> [Keras Pooling Python实现](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L707)，Keras根据其对pooling算子的需求，分别实现了四种Pooling

- AvgPool
- MaxPool
- GlobalMaxPool(GXP)
- GlobalAvgPool(GAP)

```python
# Aliases

AvgPool1D = AveragePooling1D
MaxPool1D = MaxPooling1D
AvgPool2D = AveragePooling2D
MaxPool2D = MaxPooling2D
AvgPool3D = AveragePooling3D
MaxPool3D = MaxPooling3D
GlobalMaxPool1D = GlobalMaxPooling1D
GlobalMaxPool2D = GlobalMaxPooling2D
GlobalMaxPool3D = GlobalMaxPooling3D
GlobalAvgPool1D = GlobalAveragePooling1D
GlobalAvgPool2D = GlobalAveragePooling2D
GlobalAvgPool3D = GlobalAveragePooling3D
```

## 2. [Caffe Pooling](https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto#L650-L652)
```
message PoolingParameter {
  optional Engine engine = 11 [default = DEFAULT];
  // If global_pooling then it will pool over the size of the bottom by doing
  // kernel_h = bottom->height and kernel_w = bottom->width
  optional bool global_pooling = 12 [default = false];
  // How to calculate the output size - using ceil (default) or floor rounding.
}
```
## 3. [Tensorflow GAP](https://github.com/AndersonJo/global-average-pooling)
> Tensorflow没有单独实现GAP，网友通过conv来实现
> [tf.nn.pooling](https://www.tensorflow.org/api_guides/python/nn#Pooling)中没有GAP的实现
> [tf.layers.pooling](https://www.tensorflow.org/api_docs/python/tf/layers)中没有GAP的声明
> 可以通过[tf.keras.layers.GlobalMaxPool3D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/GlobalMaxPool3D)来实现

## 4. [关于global average pooling](https://blog.csdn.net/yimingsilence/article/details/79227668)
> Global Average Pooling第一次出现在论文Network in Network中，后来又很多工作延续使用了GAP，实验证明：Global Average Pooling确实可以提高CNN效果

## 5. [caffe详解之池化层](https://xuke225.github.io/2018/01/24/DeepLearning/caffe/layer/%E6%B1%A0%E5%8C%96%E5%B1%82/)

## 6. [花式池化](https://antkillerfarm.github.io/dl/2017/08/28/Deep_Learning_13.html)
> 池化和卷积一样，都是信号采样的一种方式。
> 根据相关理论，特征提取的误差主要来自两个方面：

1. 邻域大小受限造成的估计值方差增大
2. 卷积层参数误差造成估计均值的偏移

> 一般来说，mean-pooling能减小第一种误差，更多的保留图像的背景信息，max-pooling能减小第二种误差，更多的保留纹理信息。
> Stochastic-pooling则介于两者之间，通过对像素点按照数值大小赋予概率，再按照概率进行亚采样，在平均意义上，与mean-pooling近似，在局部意义上，则服从max-pooling的准则。

## 7. [ROI Pooling](https://blog.csdn.net/lanran2/article/details/60143861)
> Fast RCNN中对于Pooling层的变种

## 8. [Spatial Pyramid Pooling](https://www.jianshu.com/p/e36aef9b7a8a)
> SPP也是一种Pooling层的变种

