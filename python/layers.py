import numpy as np

from bridge import conv2d_batch, relu_forward, leaky_relu_forward, maxpool2d_forward, matmul_bias


class Conv2D:
    def __init__(self, C_in, C_out, KH, KW, stride, pad):
        self.dW = None
        self.db = None
        self.col = None
        self.x = None
        self.fan_in = C_in * KH * KW
        self.weights = (np.random.randn(C_out, C_in, KH, KW) * np.sqrt(2.0 / fan_in)).astype(np.float32)
        self.bias = np.zeros(C_out, dtype=np.float32)
        self.stride = stride
        self.pad = pad
        self.KH = KH
        self.KW = KW

    def forward(self, x):
        self.x = x
        return conv2d_batch(x, self.weights, self.bias, self.stride, self.pad)

    def backward(self, dy):
        N, F, H_out, W_out = dy.shape

        self.db = np.sum(dy , axis = (0 , 2 , 3))  # ignoring F in shape -> shape [F]

        dy_for_dW = dy.transpose(1 , 0 , 2 , 3).reshape(F , -1)
        dy_for_dcol = dy.transpose(0 , 2 , 3 , 1).reshape(N , -1)

        dW_flat = dy_for_dW.T @ self.col.reshape(self.fan_in , -1)
        dcol = dy_for_dcol.T @ self.weights.reshape(self.fan_in, -1)



        self.dW = dW_flat.reshape(self.weights.shape)

        col2im(dcol , self.x.shape , self.KH , self.KW , self.stride, self.pad)


    def _col2im(self , dcol , input_shape , kh , kw , stride, pad):
        N, C, H, W = input_shape




class ReLU:
    def forward(self, x):
        output, self.mask = relu_forward(x)
        return output

    def backward(self, dy):
        return self.mask * dy

class LeakyReLU:
    def __init__(self, alpha):
        self.alpha = alpha

    def forward(self, x):
        output, self.mask = leaky_relu_forward(x, self.alpha)
        return output

    def backward(self, dy):
        return self.mask * dy

class MaxPool2D:
    def __init__(self, PH, PW, stride):
        self.indices = None
        self.PH = PH
        self.PW = PW
        self.stride = stride
        self.input = None

    def forward(self, x):
        self.input = x
        N, C, H_in, W_in = x.shape
        H_out = (H_in - self.PH) // self.stride + 1
        W_out = (W_in - self.PW) // self.stride + 1

        output = np.empty((N, C, H_out, W_out), dtype=np.float32)
        indices = np.empty((N, C, H_out, W_out), dtype=np.int32)

        for n in range(N):
            output[n], indices[n] = maxpool2d_forward(x[n], self.PH, self.PW, self.stride)

        self.indices = indices
        return output

    def backward(self, dy):
        N, C, H, W = self.input.shape
        dx = np.zeros((N , C*H*W) , dtype=np.float32)
        idx_flat = self.indices.reshape(N, -1) # flattening parallel views
        dy_flat = dy.reshape(N , -1)
        np.add.at(dx , (np.arange(N)[: , None] , idx_flat), dy_flat)
        return dx.reshape(N , C , H , W)


class Flatten:
    def forward(self, x):
        self.shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dy):
        return dy.reshape(self.shape)

class Dense:
    def __init__(self, in_features, out_features):
        self.weights = (np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)).astype(np.float32)
        self.bias = np.zeros(out_features, dtype=np.float32)

    def forward(self, x):
        self.cache = x
        return matmul_bias(x, self.weights, self.bias)

    def backward(self, dy):
        self.dw = self.cache.T @ dy
        self.db = np.sum(dy, axis=0)
        self.dx = dy @ self.weights.T

        return self.dx



