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
1. 依据官方推荐安装了Anaconda 3

2. create了一个conda环境，用于跑tensorflow的自定义算子。安装了tensorflow-gpu 1.8.0版本

3. conda install cython

4. export CPATH=$CPATH:/root/anaconda3/envs/tf/lib/python3.6/site-packages/external/local_config_cuda/cuda/

5. 修改文件`lib/lib_kernel/lib_psalign_pooling/make.sh`,添加 `-I /usr/local \`

```
nvcc -std=c++11 -c -g -o psalign_pooling_op.cu.o psalign_pooling_op_gpu.cu.cc \
  -I /usr/local \
  ...
```

6. 修改文件`lib/lib_kernel/lib_psalign_pooling/make.sh`

```
-D_GLIBCXX_USE_CXX11_ABI=1
0->1 或 1->0
雄哥解释是该编译选项用于控制C++11的打开和关闭
```

7. 修改文件`lib/lib_kernel/lib_psalign_pooling/make.sh`文件中`-arch参数`，解决错误`cudaCheckError() failed : no kernel image is available for execution on the device`。

```
-arch=sm_35 参数修改
```

# Reference
- [小白的tensorflow+CUDA编程踩坑记录](https://zhuanlan.zhihu.com/p/40375792)
- [CUDA arch and CUDA gencode for various NVIDIA cards](http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)
- [check CUDA version](http://arnon.dk/check-cuda-installed/)


