import numpy as np

from bridge import conv2d_batch, relu_forward, leaky_relu_forward, maxpool2d_forward, matmul_bias


class Conv2D:
    def __init__(self, C_in, C_out, KH, KW, stride, pad):
        fan_in = C_in * KH * KW
        self.weights = (np.random.randn(C_out, C_in, KH, KW) * np.sqrt(2.0 / fan_in)).astype(np.float32)
        self.bias = np.zeros(C_out, dtype=np.float32)
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        self.x = x
        return conv2d_batch(x, self.weights, self.bias, self.stride, self.pad)

class ReLU:
    def forward(self, x):
        output, self.mask = relu_forward(x)
        return output

class LeakyReLU:
    def __init__(self, alpha):
        self.alpha = alpha

    def forward(self, x):
        output, self.mask = leaky_relu_forward(x, self.alpha)
        return output

class MaxPool2D:
    def __init__(self, PH, PW, stride):
        self.indices = None
        self.PH = PH
        self.PW = PW
        self.stride = stride

    def forward(self, x):
        N, C, H_in, W_in = x.shape
        H_out = (H_in - self.PH) // self.stride + 1
        W_out = (W_in - self.PW) // self.stride + 1

        output = np.empty((N, C, H_out, W_out), dtype=np.float32)
        indices = np.empty((N, C, H_out, W_out), dtype=np.int32)

        for n in range(N):
            output[n], indices[n] = maxpool2d_forward(x[n], self.PH, self.PW, self.stride)

        self.indices = indices
        return output

class Flatten:
    def forward(self, x):
        self.shape = x.shape
        return x.reshape(x.shape[0], -1)

class Dense:
    def __init__(self, in_features, out_features):
        self.weights = (np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)).astype(np.float32)
        self.bias = np.zeros(out_features, dtype=np.float32)

    def forward(self, x):
        self.x = x
        return matmul_bias(x, self.weights, self.bias)
