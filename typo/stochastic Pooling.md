# [PyTorch](https://discuss.pytorch.org/t/is-there-any-way-to-implement-stochastic-pooling/12780)

> Hello,

> thank you for asking!

> Stochastic pooling as in the paper with stride = pool size is easy to implement using view (so that the indices to be pooled are in their own dimension e.g. x.view(x.size(0),x.size(1)//2,2)), sampling random coordinates from multinomial and using that for indexing.
The weighting can be done using a standard (“spatial”) convolution in the functional interface and a filter that contains the probability.

> You could also use stochastic average pooling by drawing scores + softmax + convolution similar to what they suggest for test time but with random weights.

> I could do an implementation example if that helps.

> Someone more knowledgable than me will have to answer fractional average pooling (fraction max pooling is done in pytorch, but I’m not aware of average pooling).
An ad-hoc way of achieving something similar could involve “splitting the activations at the border” of larger kernels by using positive convolutions stencils that sum to one but are not all equal and then keeping a subgrid.

> Best regards

> Thomas
