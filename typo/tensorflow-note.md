# tensorflow-internals阅读笔记

## 第I部分 基础知识

### 1 介绍

#### 1.1 前世今生

##### 1.1.2 TensorFlow

###### 设计原则
> TensorFlow的系统架构遵循了一些基本的设计原则，用于指导TensorFlow的系统实现。
1. 延迟计算：图的构造与执行分离，并推迟计算图的执行过程；
2. 原子OP：OP是最小的抽象计算单元，支持构造复杂的网络模型
3. 抽象设备：支持CPU， GPU, ASIC多种异构计算设备类型；
4. 抽象任务：基于任务的PS，对新的优化算法和网络模型具有良好的可扩展性。


###### Tensorflow优势
> 相对于其他机器学习框架，TensorFlow具有如下方面的优势。

1. **高性能**：TensorFlow升级至1.0版本性能提升显著，单机多卡(8卡GPU)中，Inception v3的训练实现了7.3倍的加速比；在分布式多机多卡(64卡GPU)环境中，Inception v3的训练实现了58倍的加速比；
2. **跨平台**: 支持多CPU/GPU/ASIC多种异构设备的运算；支持台式机，服务器，移动设备等多种计算平台；支持Windows，Linux，MacOS等多种操作系统；
3. **分布式**：支持本地和分布式的模型训练和推理；
4. **多语言**: 支持Python, C++, Java, Go等多种程序设计语言；
5. **通用性**: 支持各种复杂的网络模型的设计和实现，包括非前馈型神经网络；
6. **可扩展**：支持OP扩展，Kernel扩展，Device扩展，通信协议的扩展；
7. **可视化**：使用TensorBoard可视化整个训练过程，极大地降低了TensorFlow的调试过程；
8. **自动微分**：TensorFlow自动构造反向的计算子图，完成训练参数的梯度计算；
9. **工作流**：TensorFlow与TensorFlow Serving无缝集成，支持模型的训练、导入、导出、发布一站式的工作流，并自动实现模型的**热更新**和版本管理。

### 2 编程环境

#### 2.1 代码结构

##### 2.1.2 源码结构

> 本书将重点关注core，python组件，部分涉及c，cc, stream_executor组件。

##### 2.1.3 Core

> 内核的源码结构如下所示，主要包括平台，实用函数库，基础框架，Protobuf定义，本地运行时，分布式运行时，图操作，OP定义，以及Kernel实现等组成，这是本
书重点剖析的组件之一，将重点挖掘基础框架中隐藏的领域模型，追踪整个运行时环境的生命周期和图操作的详细过程，并揭示常见OP的Kernel实现原理和运行机制。

```
./tensorflow/core
```

##### 2.1.4 Python

> Python定义和实现了TensorFlow的编程模型，并对外开放API供程序员使用，其源代码如下所示。

```
./tensorflow/python
```

##### 2.1.5 Contrib

> contrib是第三方贡献的编程库，它也是TensorFlow标准化之前的实验性编程接口，犹如Boost社区与C++标准之间的关系。当contrib的接口成熟后，便会被TensorFlow标准化，并从contrib中迁移至core，python中，并正式对外发布。

##### 2.1.6 StreamExecutor

> StreamExecutor是Google另一个开源组件库，它提供了主机端(host-side)的编程模型和运行时环境，实现了CUDA和OpenCL的统一封装。使得在主机端的代码中，可以将Kernel函数无缝地部署在CUDA或OpenCL的计算设备上执行。

> 目前，StreamExecutor被大量应用于Google内部GPGPU的应用程序的运行时。其中TensorFlow运行时也包含了一个StreamExecutor的快照版本，用于封装CUDA和OpenCL的运行时。本书将简单介绍CUDA的编程模型和线程模型，并详细介绍StreamExecutor的系统架构与工作原理，揭示Kernel函数的实现模式和习惯用法。

##### 2.1.7 Compiler

> 众所周知，灵活性是TensorFlow最重要的设计目标和核心优势，因此TensorFlow的系统架构具有良好的可扩展性。TensorFlow可用于定义任意图结构，并使用异构的计算设备有效地执行。但是，熊掌与鱼翅不可兼得，当低级OP组合为计算子图时，并期望在GPU上有效执行时，运行时将启动更多的Kernel的运算。

> 因此，TensorFlow分解和组合OP的方法，在运行时并不能保证以最有效的方式运行。此时XLA技术孕育而生，它使用JIT编译计算来分析运行时的计算图，它将多个OP融合在一起，并生成更高效的本地机器代码，提升计算图的执行效率。

> XLA技术目前处于初级的研发阶段，是TensorFlow社区较为活跃研究方向，截止目前代码规模大约为12.5万行，主要使用C++实现。

