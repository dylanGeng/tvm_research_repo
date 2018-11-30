# Light-Head RCNN发布网址
> [Light-Head RCNN](https://github.com/zengarden/light_head_rcnn)

# Requiremens
> 官方推荐的安装配置如下

```
1.tensorflow-gpu==1.5.0 (We only test on tensorflow 1.5.0, early tensorflow is not supported because of our gpu nms implementation)
2.python3. We recommend using Anaconda as it already includes many common packages. (python2 is not tested)
3.Python packages might missing. pls fix it according to the error message.
```
> 目前的验证环境，在252的docker容器上搭建。

1. GPU显卡为Tesla K40m；
2. 容器为CUDA 10.0的容器。

> 安装环境搭建。
1. 依据官方推荐安装了Anaconda 3。
2. create了一个conda环境，用于跑tensorflow的自定义算子。安装了tensorflow-gpu 1.8.0版本。
3. conda install cython
