import numpy as np
import math

MAX_TYPE = "max"
AVG_TYPE = "avg"
TYPE = "type" 
PAD = "pad"
KERNEL = "kernel"
STRIDE = "stride"
INPUT_TXT = "input.txt"
OUTPUT_TXT = "output.txt"
INPUT_BIN = "input.bin"
OUTPUT_BIN = "output.bin"

def randomGaussian(size, miu = 3, sigma = 1):
    assert sigma > 0, "Error: Except positive sigma for guassaain"

    ret = np.random.normal(miu, sigma, size)
    for x in np.nditer(ret, op_flags=['readwrite']):
        if np.random.randint(0, 2):
            continue
        x[...] = x*-1
    return ret

def get_diffPad(kw, wo, w_stride, wi, pad_w):
    """
    get horizontal or vertical diff
    :param kw:
    :param wo:
    :param wstride:
    :param wi:
    :param pad_w:
    :return:
    """
    diff = kw + (wo - 1)*w_stride - (wi + pad_w + pad_w)
    if diff < 0:
        diff = 0
    return diff

def pooling_forward(x, pool_param):
    N, C1, H, W, C0 = x.shape
    pad = pool_param[PAD]
    stride = pool_param[STRIDE]
    kernel = pool_param[KERNEL]
    pad_h, pad_w = pad
    kernel_h, kernel_w = kernel
    stride_h, stride_w = stride

    ho = int(math.ceil((H + 2*pad_h - kernel_h) / float(stride_h))+1)
    wo = int(math.ceil((W + 2*pad_w - kernel_w) / float(stride_w)) + 1)

    h_end_diff = get_diffPad(kernel_h, ho, stride_h, H, pad_h)
    w_end_diff = get_diffPad(kernel_w, wo, stride_w, W, pad_w)

    x_padded = -65535.00*np.ones((N, C1, H+2*pad_h+h_end_diff, W+2*pad_w+w_end_diff, C0))
    x_padded[:, :, pad_h:pad_h+H, pad_w:pad_w+W, :] = x

    out = np.zeros((N, C1, ho, wo, C0))

    #print(x.shape)
    #print(x)
    #print(x_padded.shape)
    #print(x_padded)
    for i in range(ho):
        for j in range(wo):
            out[:, :, i, j, :] = np.max(x_padded[:, :, i*stride_h:i*stride_h+kernel_h, j*stride_w:j*stride_w+kernel_w, :], axis=(2, 3))
    #cache = ()
    #print(out)
    return out


def gen_data(shape, kernel=None, pad=None, stride=None):
    x = randomGaussian(shape, miu=1, sigma=0.1).astype(np.float16)

    pool_param = {KERNEL: kernel, PAD: pad, STRIDE: stride, TYPE: MAX_TYPE}
    out = pooling_forward(x, pool_param)

    x_num = reduce(lambda i, j: i * j, shape[:])
    out_num = reduce(lambda i, j: i * j, out.shape[:])

    print ("shape of x is : " + str(x.shape))
    print ("length of x is : " + str(x_num))
    print ("shape of out is : " + str(out.shape))
    print ("length of out is : " + str(out_num))

    np.savetxt(INPUT_TXT, x.reshape(x_num))
    print ("Save " + INPUT_TXT + " successfully!")
    np.savetxt(OUTPUT_TXT, out.reshape(out_num))
    print ("Save " + OUTPUT_TXT + " successfully!")

    with open(INPUT_BIN, 'wb') as fo:
        fo.write(x.astype(np.float16, copy=False))
        print ("Write " + INPUT_BIN + " successfully!")
    with open(OUTPUT_BIN, 'wb') as fo:
        fo.write(out.astype(np.float16, copy=False))
        print ("Write " + OUTPUT_BIN + " successfully!")

if __name__ == "__main__":
    shape_ = (1, 4, 112, 112, 16)
    kernel_ = (3, 3)
    pad_ = (1, 1)
    stride_ = (2, 2)
    print(KERNEL + " is : " + str(kernel_))
    print(STRIDE + " is : " + str(stride_))
    print(PAD + " is : " + str(pad_))
    gen_data(shape=shape_, kernel=kernel_, pad=pad_, stride=stride_)
