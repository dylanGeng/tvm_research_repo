# C++11新特性: Lambda表达式

> 或许，Lambda表达式算得上是C++11新特性中最激动人心的一个。这个全新的特性听起来很深奥，但却是很多其他语言早已提供(比如C#)或者即将提供(比如Java)的。简而言之，Lambda表达式就是用于创建匿名函数的。GCC 4.5.x和Microsoft Visual Studio早已提供了对Lambda表达式的支持。在GCC 4.7中，默认是不开启C++11特性的，需要添加-std=c++11编译参数。而VS2010则默认开启。

> 为什么说lambda表达式如此激动人心呢？举一个例子。标准C++库中有一个常用算法的库，其中提供了很多算法函数，比如sort()和find()。这些函数通常需要提供一个“谓词函数predicate function”。所谓谓词函数，就是进行一个操作用的临时函数。比如find()需要一个谓词，用于查找元素满足的条件；能够满足谓词函数的元素才会被查找出来。这样的谓词函数，使用临时的匿名函数，既可以减少函数数量，又会让代码变得清晰易读。

```
#include <algorithm>
#include <cmath>

void abssort(float *x, unsigned N)
{
  std::sort(x,
            x + N,
            [](float a, float b) { return std::abs(a) < std::abs(b); });
}
```

> 从上面的例子来看，尽管支持lambda表达式，但C++的语法看起来却很“神奇”。lambda表达式使用一对方括号作为开始的标识，类似于声明一个函数，只不过这个函数没有名字，也就是一个匿名函数。这个匿名函数接受两个参数，a和b；其返回值是一个bool类型的值，注意，返回值是自动推断的，不需要显式声明，不过这是有条件的！条件就是lambda表达式的语句只有一个return。函数的作用是比较a\b的绝对值的大小。然后，在此例中，这个lambda表达式作为一个闭包被传递给std::sort()函数。

> 下面，我们来详细解释下这个神奇的语法到底代表着什么。

> 我们从另外一个例子开始：

```
std::cout << [](float f) { return std::abs(f); } (-3.5);
```

> 输出值是什么？3.5!注意，这是一个函数对象(由lambda表达式生成)，其实参是-3.5，返回值是参数的绝对值。Lambda表达式的返回值类型是语言自动推断的，因为std::abs()的返回值就是float。注意，前面我们也提到了，只有当lambda表达式中的语句"足够简单"，才能自动推断返回值类型。

> C++11的这种语法，其实就是匿名函数声明之后马上调用(否则的话，如果这个匿名函数既不调用，又不作为闭包传递给其他函数，那么这个匿名函数就没有什么用处)。如果你觉得奇怪，那么来看看JavaScript的这种写法：

```
function() {} ();

function(a) {} (-3.5);
```

> C++11的写法完全类似JavaScript的语法。

> 如果我不想让lambda表达式自动推断类型，或者是lambda表达式的内容很复杂，不能自动推断怎么办？比如，std::abs(float)的返回值是 float，我想把它强制转型为 int。那么，此时，我们就必须显式指定 lambda 表达式返回值的类型：

```
std::cout << [](float f) -> int { return std::abs(f); } (-3.5);
```

> 这个语句与前面的不同之处在于，lambda 表达式的返回时不是 float 而是 int。也就是说，上面语句的输出值是 3。返回值类型的概念同普通的函数返回值类型是完全一样的。

> 当我们想引用一个 lambda 表达式时，我们可以使用auto关键字，例如：
```
auto lambda = [] () -> int { return val * 100; };
```
> auto关键字实际会将 lambda 表达式转换成一种类似于std::function的内部类型（但并不是std::function类型，虽然与std::function“兼容”）。所以，我们也可以这么写：
```
std::function<int()> lambda = [] () -> int { return val * 100; };
```
> 如果你对std::function<int()>这种写法感到很神奇，可以查看 C++ 11 的有关std::function的用法。简单来说，std::function<int()>就是一个可调用对象模板类，代表一个可调用对象，接受 0 个参数，返回值是int。所以，当我们需要一个接受一个double作为参数，返回int的对象时，就可以写作：std::function<int(double)>。

> 引入 lambda 表达式的前导符是一对方括号，称为 lambda 引入符（lambda-introducer）。lambda 引入符是有其自己的作用的，不仅仅是表明一个 lambda 表达式的开始那么简单。lambda 表达式可以使用与其相同范围 scope 内的变量。这个引入符的作用就是表明，其后的 lambda 表达式以何种方式使用（正式的术语是“捕获”）这些变量（这些变量能够在 lambda 表达式中被捕获，其实就是构成了一个闭包）。目前为止，我们看到的仅仅是一个空的方括号，其实，这个引入符是相当灵活的。例如：
```
float f0 = 1.0;
std::cout << [=](float f) { return f0 + std::abs(f); } (-3.5);
```
> 其输出值是 4.5。[=] 意味着，lambda 表达式以传值的形式捕获同范围内的变量。另外一个例子：
```
float f0 = 1.0;
std::cout << [&](float f) { return f0 += std::abs(f); } (-3.5);
std::cout << '\n' << f0 << '\n';
```
> 输出值是 4.5 和 4.5。[&] 表明，lambda 表达式以传引用的方式捕获外部变量。那么，下一个例子：
```
float f0 = 1.0;
std::cout << [=](float f) mutable { return f0 += std::abs(f); } (-3.5);
std::cout << '\n' << f0 << '\n';
```
> 这个例子很有趣。首先，[=]意味着，lambda 表达式以传值的形式捕获外部变量。C++ 11 标准说，如果以传值的形式捕获外部变量，那么，lambda 体不允许修改外部变量，对 f0 的任何修改都会引发编译错误。但是，注意，我们在 lambda 表达式前声明了mutable关键字，这就允许了 lambda 表达式体修改 f0 的值。因此，我们的例子本应报错，但是由于有 mutable 关键字，则不会报错。那么，你会觉得输出值是什么呢？答案是，4.5 和 1.0。为什么 f0 还是 1.0？因为我们是传值的，虽然在 lambda 表达式中对 f0 有了修改，但由于是传值的，外部的 f0 依然不会被修改。
> 上面的例子是，所有的变量要么传值，要么传引用。那么，是不是有混合机制呢？当然也有！比如下面的例子：
```
float f0 = 1.0f;
float f1 = 10.0f;
std::cout << [=, &f0](float a) { return f0 += f1 + std::abs(a); } (-3.5);
std::cout << '\n' << f0 << '\n';
```
> 这个例子的输出是 14.5 和 14.5。在这个例子中，f0 通过引用被捕获，而其它变量，比如 f1 则是通过值被捕获。
> 下面我们来总结下所有出现的 lambda 引入符：
- []        // 不捕获任何外部变量
- [=]      // 以值的形式捕获所有外部变量
- [&]      // 以引用形式捕获所有外部变量
- [x, &y] // x 以传值形式捕获，y 以引用形式捕获
- [=, &z]// z 以引用形式捕获，其余变量以传值形式捕获
- [&, x]  // x 以值的形式捕获，其余变量以引用形式捕获

> 另外有一点需要注意。对于[=]或[&]的形式，lambda 表达式可以直接使用 this 指针。但是，对于[]的形式，如果要使用 this 指针，必须显式传入：
```
[this]() { this->someFunc(); }();
```
> 至此，我们已经大致了解了 C++ 11 提供的 lambda 表达式的概念。建议通过结合 lambda 表达式与std::sort()或std::for_each()这样的标准函数来尝试使用一下吧！

## [将Lambda传递到函数指针](https://www.oracle.com/technetwork/cn/articles/servers-storage-dev/howto-use-lambda-exp-cpp11-2189895-zhs.html)

> C++11标准库中有一个名为function的模板，它可以接受指定类型的函数或者具有匹配的返回类型和参数列表的lambda。这将产生一个指向函数类型的指针，例如，清单4可用作函数参数类型，接受int参数，返回void。您可以向其传递任何类似匹配函数或lambda 的内容。

```
清单4
std::function<void(int)>
```

> 清单5显示的函数扫描一个数组，对每个元素应用一个给定函数。

```
清单5
void scan( int* a, int length, std::function<void(int)> process )
{
  for(int i=0; i<length; i++) {
      process(a[i]);
  }
}
```
> 清单6显示如何通过传递一个函数或lambda 表达式作为参数来调用scan() 函数。

```
清单6
void f(int);
int a[10];
...
scan(a, 10, f);
scan(a, 10, [](int k)->void { ... } );
```

## [Lambda表达式中的变量捕获](https://www.oracle.com/technetwork/cn/articles/servers-storage-dev/howto-use-lambda-exp-cpp11-2189895-zhs.html)

> 到目前为止，我们对 lambda 表达式的处理基本与标准函数调用类似：传入参数，返回结果。然而，在函数主体中声明的 lambda 表达式还是可以捕获在声明 lambda 处可见的函数的任何局部变量。

> 假设我们需要使用函数 scan()，但希望 process 函数只对大于某个阈值的值起作用。我们不能修改 scan()，不能让 scan() 向 process 函数传递多个参数。但如果我们将一个 lambda 表达式传递给 scan() 函数，则可以从其环境捕获一个局部变量。

> 在清单7中，我们将希望捕获的变量放在方括号中，即放在捕获表达式中。这实际上向 lambda 表达式中额外传递了一个参数，但无需更改 scan 函数的定义。就像传递参数给函数一样，我们实际上是在函数的调用点捕获值 threshold 的副本，这称为通过值捕获。

```
清单7
#include <algorithm>
void scan( int* a, int length, std::function<void(int)> process)
{
  for(int i=0; i<length; i++) {
    process(a[i]);
  }
}
int main()
{
  int a[10] = { 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 };
  int threshold = 5;
  scan(a, 10,
    [threshold](int v)
    { if (v>threshold) { printf("%i ", v); } }
  );
  printf("\n");
  return 0;
}
```
> 有一个简写形式 [=]，表示“通过值捕获每个变量”。在清单 8 中，我们将函数调用重新编写为使用这种更短的表达式。

```
清单8
scan(a, 10, [=](int v) { if (v>threshold) { printf("%i ", v); } });
```

> 注：通过值捕获变量意味着生成局部副本。如果有多个局部变量，全部捕获可能会导致 lambda 产生显著开销。

> 但有些情况下，我们希望修改捕获的变量。例如，假设我们要计算最大值并将其存储在变量 max 中。在这种情况下，我们不想使用该变量值的副本，而是希望使用该变量的引用，这样，我们就可以在模板中修改该变量。这称为通过引用捕获变量。清单 9 显示了这样一个示例。

```
清单9
#include <algorithm>
void scan(int * a, int length, std::function<void (int)> func)
{
  for(int i=0; i<length; i++) {
    func(a[i]);
  }
}

int main()
{
  int a[10] = { 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 };
  int threshold = 5;
  int max =0;
  std::sort( a, &a[10], [](int x, int y){return (x < y);});
  scan(a, 10,
   [threshold,&max](int v) { if (v>max) {max = v;}
                     if (v>threshold) { printf("%i ", v); } });
  printf("\n");
  printf("Max = %i\n",max);
  return 0;
}
```

> 同样，也有一个简写形式 [&]，用于应通过引用捕获每个变量的情况。

## [Lambda表达式、函数对象和函子](https://www.oracle.com/technetwork/cn/articles/servers-storage-dev/howto-use-lambda-exp-cpp11-2189895-zhs.html)

> 虽然lambda表达式是C++11的新特性，但用这种方式访问现有语言特性的确很方便。lambda表达式是函数对象的速记表示法。函数对象是一个具有成员operator()()(函数调用运算符)的类类型对象，因此可以像函数一样调用。函数对象类型被称作**函子**。清单10显示了一个函子的示例。

```
清单10
class compare_ints {
public:
  compare_ints(int j, int k ) : l(j), r(k) { }
  bool operator()() { return l < r; }
private:
  int l, r;
};
```

> 您可以创建一个`compare_ints`对象，用两个整型值初始化，如果第一个值小于第二个值，使用函数调用运算符返回true：

```
compare_ints comp(j, k);
bool less_than = comp();
```

> 也可以动态创建一个临时对象，然后直接使用：

```
bool less_than = compare_ints(j, k)();
```

> 使用lambda表达式不必创建和命名函子类即可达到这种效果，编译器为您创建一个匿名函子，如清单11所示。

```
清单11
auto comp = [](int i, int j){ return j < k; };
bool less_than = comp(l, r);
```

> 在清单11中，comp是匿名函子类型的对象。

> 您也可以动态执行此操作：

```
bool less_than = [l,r]() { return l < r; }();
```

## 总结

> Lambda表达式是一种非常强大的C++扩展。它们不仅具有函数指针的灵活性，还可以通过捕获局部变量提高可扩展性。

> 显然，与C++11中广泛的模板特性结合时，lambda表达式会变得更加有用，这种情况在按C++11标准编写的代码中会经常遇到。

# Reference
1. [参考文章](https://blogs.oracle.com/pcarlini/entry/c_1x_tidbits_lambda_expressions)
2. [如何在C++11中使用Lambda表达式](https://www.oracle.com/technetwork/cn/articles/servers-storage-dev/howto-use-lambda-exp-cpp11-2189895-zhs.html)
