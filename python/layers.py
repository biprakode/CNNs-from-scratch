import numpy as np

from bridge import conv2d_batch, im2col, relu_forward, leaky_relu_forward, maxpool2d_forward, matmul_bias


def col2im(dcol, input_shape, KH, KW, stride, pad):
    # Inverse of im2col: scatter each row of dcol back into the input-gradient
    # tensor. Overlapping patches MUST accumulate (+=). Using = would silently
    # drop gradients where receptive fields overlap.
    #   dcol:        [N*H_out*W_out, C*KH*KW]
    #   input_shape: (N, C, H, W)
    #   returns dx:  [N, C, H, W]
    N, C, H, W = input_shape
    H_pad = H + 2 * pad
    W_pad = W + 2 * pad
    H_out = (H_pad - KH) // stride + 1
    W_out = (W_pad - KW) // stride + 1

    # Scatter into padded scratch; slice the border off at the end. This removes
    # bounds checks inside the inner loop.
    dx_padded = np.zeros((N, C, H_pad, W_pad), dtype=np.float32)

    # Reshape dcol so each patch row lays out as (C, KH, KW) — exactly the shape
    # of the slice we'll write into dx_padded.
    dcol = dcol.reshape(N, H_out, W_out, C, KH, KW)

    for n in range(N):
        for oh in range(H_out):
            for ow in range(W_out):
                ih0 = oh * stride
                iw0 = ow * stride
                # += is the whole point of col2im.
                dx_padded[n, :, ih0:ih0 + KH, iw0:iw0 + KW] += dcol[n, oh, ow]

    if pad == 0:
        return dx_padded
    return dx_padded[:, :, pad:pad + H, pad:pad + W]


class Conv2D:
    def __init__(self, C_in, C_out, KH, KW, stride, pad):
        self.dW = None
        self.db = None
        self.col = None
        self.x = None
        self.fan_in = C_in * KH * KW
        self.weights = (np.random.randn(C_out, C_in, KH, KW) * np.sqrt(2.0 / self.fan_in)).astype(np.float32)
        self.bias = np.zeros(C_out, dtype=np.float32)
        self.stride = stride
        self.pad = pad
        self.KH = KH
        self.KW = KW

    def forward(self, x):
        # Cache input for backward. We also need the im2col matrix (the
        # "flattened patches" matrix): conv2d_batch builds it internally but
        # doesn't return it, so rebuild it here per-image and stack.
        # Shape: [N, H_out*W_out, C*KH*KW]. Each row = one flattened patch.
        self.x = x
        N = x.shape[0]
        self.col = np.stack([im2col(x[n], self.KH, self.KW, self.stride, self.pad) for n in range(N)])
        return conv2d_batch(x, self.weights, self.bias, self.stride, self.pad)

    def backward(self, dy):
        # dy: [N, F, H_out, W_out] — upstream gradient.
        # Populates self.dW [F, C, KH, KW] and self.db [F]. Returns dx [N, C, H, W].
        N, F, H_out, W_out = dy.shape

        # --- db: bias gradient ---
        # b[f] adds to every (n, oh, ow) cell for filter f, so db[f] is dy summed
        # over N, H_out, W_out; F stays.
        self.db = np.sum(dy, axis=(0, 2, 3))  # [F]

        # --- Two dy reshapes, one per matmul ---
        # For dW we need rows = filter, cols = patch index:
        #   [N, F, H_out, W_out] --transpose(1,0,2,3)--> [F, N, H_out, W_out]
        #                        --reshape(F, -1)-----> [F, N*H_out*W_out]
        dy_for_dW = dy.transpose(1, 0, 2, 3).reshape(F, -1)

        # For dcol we need rows = patch index, cols = filter:
        #   [N, F, H_out, W_out] --transpose(0,2,3,1)--> [N, H_out, W_out, F]
        #                        --reshape(-1, F)-----> [N*H_out*W_out, F]
        dy_for_dcol = dy.transpose(0, 2, 3, 1).reshape(-1, F)

        # --- dW: filter gradient ---
        # Flatten col [N, H_out*W_out, C*KH*KW] -> [N*H_out*W_out, C*KH*KW] so
        # the patch-index axis aligns with dy_for_dW's second axis.
        col_flat = self.col.reshape(-1, self.fan_in)

        # [F, N*H_out*W_out] @ [N*H_out*W_out, C*KH*KW] -> [F, C*KH*KW]
        # Then reshape back to the weights' 4D layout.
        dW_flat = dy_for_dW @ col_flat
        self.dW = dW_flat.reshape(self.weights.shape)

        # --- dcol: gradient wrt the (intermediate) im2col matrix ---
        W_flat = self.weights.reshape(F, -1)  # [F, C*KH*KW]

        # [N*H_out*W_out, F] @ [F, C*KH*KW] -> [N*H_out*W_out, C*KH*KW]
        dcol = dy_for_dcol @ W_flat

        # --- dx: scatter dcol back into input-shaped gradient via col2im ---
        dx = col2im(dcol, self.x.shape, self.KH, self.KW, self.stride, self.pad)
        return dx




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



