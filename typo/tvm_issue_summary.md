# tvm issue



## [Supporting reduction domains where the RDom depends on axis variable](https://github.com/dmlc/tvm/issues/2207)

```python
from topi.util import get_const_tuple
import tvm

def func(Elements, Lengths):
    def f(n, d):
        rg = tvm.reduce_axis((0, Lengths[n]))
        return tvm.sum(Elements[rg, d], axis=rg)

    (N,) = get_const_tuple(Lengths.shape)
    (_, D) = get_const_tuple(Elements.shape)
    return tvm.compute((N, D), f, name="Y")

def run(N, I, D):
    Elements = tvm.placeholder(shape=(I, D), dtype="float32", name="Elements")
    Lengths = tvm.placeholder(shape=(N,), dtype="int32", name="Lengths")
    Y = func(Elements, Lengths)
    s = tvm.create_schedule([Y.op])
    print(tvm.lower(s, [Elements, Lengths, Y], simple_mode=True))
    print(tvm.save_json(Y))
    f = tvm.build(s, [Elements, Lengths, Y], target="llvm")

run(N=10, I=10, D=128)
```

