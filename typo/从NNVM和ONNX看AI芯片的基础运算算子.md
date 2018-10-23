# [从NNVM和ONNX看AI芯片的基础运算算子](https://zhuanlan.zhihu.com/p/32711259)

> NNVM是由陈天奇团队提出的一套可复用的计算流图中间表达层，它提供了一套精简的API函数，用以构建、表达和传输计算流图，从而便于高层级优化。另外NNVM也可以作为多个深度学习框架的共享编译器，可以优化、编译和部署在多种不同的硬件后端。其特点是部署的模型拥有最小依赖，可加入新的操作(operators)，可将新的优化通路加入到现有的图结构中。从NNVM的观点看，它可用于将M个框架，N个机器之间构建一个单一的纽带，以实现各种框架向各种实现平台的无差别部署。

![NNVM](https://pic3.zhimg.com/80/v2-18d0443d567986dc4f34d23e4daa890d_hd.jpg)

> [ONNX](https://onnx.ai/)是Facebook联合微软和AWS推出的开源的深度学习表示格式。通过ONNX，AI开发人员可以容易地在不同模型和工具间转换，并将工具组合使用。目前可以支持Caffe2, CNTK，PyTorch等主流模型，也对Tensorflow提供了早期支持。以下是ONNX的一些基本信息。

![ONNX](https://pic2.zhimg.com/80/v2-3c2b014d32803c7694c16543c9350458_hd.jpg)
![framework](https://pic3.zhimg.com/80/v2-1535e9aa53ca07773a8daf87a25714de_hd.jpg)
![Hardware Optimizations](https://pic3.zhimg.com/80/v2-aa65dfafbc870bbdad893feb005873d6_hd.jpg)

> 严格而言，和NNVM相比，ONNX更像是一个协议转换器，可以在各个框架之间进行转换。

> 目前多种标准的转换工具已经开发了出来，如表格所示。

![Framework/tool](https://pic1.zhimg.com/80/v2-693ed13472a588b979ad7c7f963cc919_hd.jpg)

> 在此，我们将借用NNVM和ONNX的算子，分析AI硬件加速的需求。这些算子都包含了相关参数，在此不细致表述。有关这些算子的完整解释，可以参看[ONNX的官方表述](https://github.com/onnx/onnx/blob/master/docs/Operators.md)；NNVM目前的唯一版本是0.8，因此本文参看版本0.8。[算子的官方解释参见](https://docs.tvm.ai/nnvm_top.html)。

> ONNX以张量(Tensor)作为输入输出数据格式，可以认为是多维数组。主要支持如下的数据格式：tensor(float16)，tensor(float)，tensor(double)。目前还不太支持定点，但提供了定点化的一些函数，如floor，ceil，clip，cast等。NNVM也以Tensor作为数据格式。

> ONNX不完整支持神经网络的训练，但它提供了训练所需要的完整的图描述，其实就是对BN和dropout两个功能上加了一个区别参数，用来描述可训练的图和不可训练的图。由于ONNX实际上就是把各种框架的图转换成了它自己的Operator表示的图，它只负责描述这个图的结构，具体前后向的计算都需要一个执行框架(称为后端)。因此，如果需要实现训练，需要实现系统根据这个图结构推演出反向图的各个步骤，这个过程可以是工具链自动的。而对于前向而言，基本不需要转化就可以直接部署在支持这些算子的实现平台上。

> 本人对这些算子进行了一个简单的归类，并将ONNX和NNVM进行了比对。由于在硬件实现上，不同的算子的实现复杂度是不同的，因此加入了Complexity的度量。另外，根据当前神经网络在图像/语言/文本三方面的应用情况，对这些算子的使用频率进行了评估。由于应用领域和硬件平台各不相同，因此复杂度和使用频率仅做参考。

# 1. 深度神经网络计算

## 1.1 计算层
> 这部分算子是深度网络的核心，用于将输入的神经元激活值与突触连接强度(权重)进行积分求和，得到新的神经元的模电位。根据是否滑窗，是否具有时序结构，可分为如下几种算子，其中FC(全连接)是多层感知机(MLP)的基础，Conv和FC是深度卷积神经网络的基础。RNN, GRU, LSTM是带有时序结构的神经网络模型，主要用于非静态图像的场合，例如语音/文字/视频等。可见，ONNX的关注比较全面，包括了时序模型，而NNVM暂时还没有包括时序模型。

![ONNX Operator](https://pic1.zhimg.com/80/v2-78d41e07483134a1becd55498154e788_hd.jpg)

> 注:代表ONNX库中此函数带有实验阶段(Experimental)标志。下同。

## 1.2 池化层

> 池化层主要用于尺度变换，提取高纬特征。主要分三种池化，


