
> 将c程序编译成so文件

'''
$ gcc c_program.c -fPIC -shared -o c_program.so
'''

> 在当前目录下会产生一个c_program.so文件

> 其中 -fPIC是position independent code(位置无关代码)的意思

> -shared是产生一个可以与其他对象连接来形成一个可执行文件的共享对象的一个参数


1. [python调用so文件](https://mp.weixin.qq.com/s?__biz=MzIxNTUzNjMyNA==&mid=2247483731&idx=1&sn=66b3b10a9efffb87c33b1bd2a077c461&chksm=979786f8a0e00fee7e5238c93442b07b92464868b96e9bbf45b6c9a067d3edff9a55f4466603#rd)
