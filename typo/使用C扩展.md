# [使用C扩展](http://wiki.jikexueyuan.com/project/interpy-zh/c_extensions/README.html)

> CPython还为开发者实现了一个有趣的特性，使用Python可以轻松调用C代码

> 开发者有三种方法可以在自己的Python代码中来调用C编写的函数-ctypes，SWIG，Python/C API。每种方式也都有各自的利弊。

> 首先，我们要明确为什么要在Python中调用C？

> 常见原因如下：

- 你要提升代码的运行速度，而且你知道C要比Python快50倍以上
- C语言中有很多传统类库，而且有些正是你想要的，但你又不想用Python去重写它们
- 想对从内存到文件接口这样的底层资源进行访问
- 不需要理由，就是想这样做

## [CTypes](http://wiki.jikexueyuan.com/project/interpy-zh/c_extensions/ctypes.html)

> Python中的ctypes模块可能是Python调用C方法中最简单的一种。ctypes模块提供了和C语言兼容的数据类型和函数来加载dll文件，因此在调用时不需对源文件做
任何的修改。也正是如此奠定了这种方法的简单性。

> 示例如下

> 实现两数求和的C代码，保存为add.c
```
//sample C file to add 2 numbers - int and floats

#include <stdio.h>

int add_int(int, int);
float add_float(float, float);

int add_int(int num1, int num2){
    return num1 + num2;

}

float add_float(float num1, float num2){
    return num1 + num2;

}
```
> 接下来将C文件编译为.so文件(windows下为DLL)。下面操作会生成adder.so文件
```
#For Linux
$  gcc -shared -Wl,-soname,adder -o adder.so -fPIC add.c

#For Mac
$ gcc -shared -Wl,-install_name,adder.so -o adder.so -fPIC add.c
```
> 现在在你的Python代码中来调用它
```
from ctypes import *

#load the shared object file
adder = CDLL('./adder.so')

#Find sum of integers
res_int = adder.add_int(4,5)
print "Sum of 4 and 5 = " + str(res_int)

#Find sum of floats
a = c_float(5.5)
b = c_float(4.1)

add_float = adder.add_float
add_float.restype = c_float
print "Sum of 5.5 and 4.1 = ", str(add_float(a, b))
输出如下

Sum of 4 and 5 = 9
Sum of 5.5 and 4.1 =  9.60000038147
```
> 在这个例子中，C文件是自解释的，它包含两个函数，分别实现了整形求和和浮点型求和。

> 在Python文件中，一开始先导入ctypes模块，然后使用CDLL函数来加载我们创建的库文件。这样我们就可以通过变量adder来使用C类库中的函数了。
当adder.add_int()被调用时，内部将发起一个对C函数add_int的调用。ctypes接口允许我们在调用C函数时使用原生Python中默认的字符串型和整型。

> 而对于其他类似布尔型和浮点型这样的类型，必须要使用正确的ctype类型才可以。如向adder.add_float()函数传参时, 我们要先将Python中的十进制值转化为
c_float类型，然后才能传送给C函数。这种方法虽然简单，清晰，但是却很受限。例如，并不能在C中对对象进行操作。

## [SWIG](http://wiki.jikexueyuan.com/project/interpy-zh/c_extensions/swig.html)
> SWIG是Simplified Wrapper and Interface Generator的缩写。是Python中调用C代码的另一种方法。在这个方法中，开发人员必须编写一个额外的接口文件来作为
SWIG(终端工具)的入口。

> Python开发者一般不会采用这种方法，因为大多数情况它会带来不必要的复杂。而当你有一个C/C++代码库需要被多种语言调用时，这将是个非常不错的选择。

> 示例如下(来自SWIG官网)
> example.c文件中的C代码包含了不同的变量和函数
```
#include <time.h>
double My_variable = 3.0;

int fact(int n) {
    if (n <= 1) return 1;
    else return n*fact(n-1);

}

int my_mod(int x, int y) {
    return (x%y);

}

char *get_time()
{
    time_t ltime;
    time(&ltime);
    return ctime(&ltime);

}
```

> 编译它

```
unix % swig -python example.i
unix % gcc -c example.c example_wrap.c \
    -I/usr/local/include/python2.1
unix % ld -shared example.o example_wrap.o -o _example.so
```

> 最后，Python的输出
```
>>> import example
>>> example.fact(5)
120
>>> example.my_mod(7,3)
1
>>> example.get_time()
'Sun Feb 11 23:01:07 1996'
>>>
```
> 我们可以看到，使用SWIG确实达到了同样的效果，虽然下了更多的工夫，但如果你的目标是多语言还是很值得的。

## [Python/C API]()

