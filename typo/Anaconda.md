# conda的环境管理

## Anaconda环境的创建

```
conda create -n py3 python=3.6
```

> 其中py3表示创建环境的名字，后面python=3.5表示创建的版本。

```
conda create -n py3 python=3.5 numpy pandas
```

> 这个是在创建环境的时候同时安装包

## Anaconda环境的激活

### OSX/Linux上
```
source activate py3
```
> py3为环境名，上述表示激活py3

### windows下
```
activate py3
```

## Anaconda环境管理
> 列出所有环境
```
conda env list
```
> 删除环境
```
conda env remove -n py3
```
> 上述表示删除环境名为py3的环境

# conda的包管理

> conda的包管理比较好理解，功能与pip类似。

## conda的一些常用操作

```
# 查看当前环境下已安装的包
conda list

# 查看某个指定环境的已安装包
conda list -n python34

# 查找package信息
conda search numpy

# 安装package
conda install -n python34 numpy
# 如果不用-n指定环境名称，则被安装在当前活跃环境
# 也可以通过-c指定通过某个channel安装

# 更新package
conda update -n python34 numpy

# 删除package
conda remove -n python34 numpy
```
> conda将conda、python都视为package，因此，完全可以使用conda来管理conda和python的版本，例如
```
# 更新conda，保持conda最新
conda update conda

# 更新anaconda
conda update anaconda

# 更新python
conda update python
# 假设当前环境是python 3.4, conda会将python升级为3.4.x系列的当前最新版本
```

> 补充：如果创建新的python环境，比如3.4，运行conda create -n python34 python=3.4之后，conda仅安装python 3.4相关的必须项，如python, pip等，如果希望该环境像默认环境那样，安装anaconda集合包，只需要：

```
# 在当前环境下安装anaconda包集合
conda install anaconda

# 结合创建环境的命令，以上操作可以合并为
conda create -n python34 python=3.4 anaconda
# 也可以不用全部安装，根据需求安装自己需要的package即可
```
# 设置国内镜像
> 如果需要安装很多packages，你会发现conda下载的速度经常很慢，因为Anaconda.org的服务器在国外。所幸的是，清华TUNA镜像源有Anaconda仓库的镜像，我们将其加入conda的配置即可：
```
# 添加Anaconda的TUNA镜像
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
# TUNA的help中镜像地址加有引号，需要去掉

# 设置搜索时显示通道地址
conda config --set show_channel_urls yes
```

> 执行完上述命令后，会生成`~/.condarc(Linux/Mac)`或`C:\Users\USER_NAME\.condarc`文件，记录着我们对conda的配置，直接手动创建、编辑该文件是相同的效果


# Reference
- [Anaconda使用总结](https://www.jianshu.com/p/2f3be7781451)
