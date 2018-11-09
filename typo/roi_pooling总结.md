## ROI Pooling层
> caffe prototxt定义：
```
layer {
  name: "roi_pool5"
  type: "ROIPooling"
  bottom: "conv5"
  bottom: "rois"
  top: "pool5"
  roi_pooling_param {
    pooled_w: 6
    pooled_h: 6
    spatial_scale: 0.0625 # 1/16
  }
}
```

> caffe caffe.proto ROI Pooling层参数说明：
```
optional ROIPoolingParameter roi_pooling_param = 43;

 message ROIPoolingParameter {
// Pad, kernel size, and stride are all given as a single value for equal
// dimensions in height and width or as Y, X pairs.
  optional uint32 pooled_h = 1 [default = 0]; // The pooled output height 池化后输出的 height
  optional uint32 pooled_w = 2 [default = 0]; // The pooled output width  池化后输出的 width
  // Multiplicative spatial scale factor to translate ROI coords from their
  // input scale to the scale used when pooling
  // 乘以空间缩放因子，以将 RoI 坐标由输入尺度转换到 pooling 时使用的尺度
  optional float spatial_scale = 3 [default = 1];
}
```

> 根据 prototxt 定义可以看出，roi_pool5 的输入有两个，bottom[0] 是 conv5 卷积层出来的 feature map，由于前面进行的 pool 层，conv5 的 feature map
的 height 和 width 分别是原图尺寸的 1/16. bottom[1] 是 rois blob， 其类似于一个 num_rois×5num_rois×5 的二维矩阵，行数 num_rosi 为
bottom[1]->num()，列数为 5，其定义为：

```
input: "rois"
input_shape {
  dim: 1 # to be changed on-the-fly to num ROIs
  dim: 5 # [batch_index, x1, y1, x2, y2] zero-based indexing
}
```

> bottom_index为每次在bottom[0]中的第一个维度的偏移，[x1,y1,x2,y2]是feature map中点的坐标。

> 在feature map中，ROI Pooling层首先计算定义的rois的conv feature map上所映射的两个坐标--(x1*spatial_scale, y1*spatial_scale)，
(x2*spatial_scale, y2*spatial_scale)，对应点为(top-left, bottom-right)，即在feature map中确定一个区域。

> 对于确定的一个区域，进行pooled_h*pooled_w(这里是6*6)等分，划分为36个相同大小的子区域，区域大小为bin_h = roi_h / pooled_h, 
bin_w = roi_w/pooled_w;

> 对于每个子区域，采用max操作找出对应feature map的最大值，即为输出top blob的对应值。

> 对于bottom[0]的每个channel进行相同操作。

> roi_pool5有一个输出top[0]，其尺寸为(bottom[1]->num(),bottom[0]->channels,pooled_h,pooled_w)，其中，pooled_h和pooled_w是固定定义的，其值这里
为6.


## Reference
1. [Caffe源码-ROI Pooling层](https://blog.csdn.net/zziahgf/article/details/78330085)
2. [ROI pooling in TensorFlow](https://github.com/deepsense-ai/roi-pooling)
