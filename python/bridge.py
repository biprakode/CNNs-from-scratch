import ctypes, os
import operator

import numpy as np
from functools import reduce

lib = ctypes.cdll.LoadLibrary("/run/media/biprarshi/COMMON/files/AI/CNNs_from_scratch/build/libcnn.so")

c_float_p = ctypes.POINTER(ctypes.c_float)
c_int_p = ctypes.POINTER(ctypes.c_int)

lib.matmul.argtypes = [
    c_float_p, c_float_p, c_float_p, # A, B, C (float pointers)
    ctypes.c_int, ctypes.c_int, ctypes.c_int, # M, K, N
    ctypes.c_int, ctypes.c_int # trans_a, trans_b
]
lib.matmul.restype = None

def matmul(A , B , trans_a=0 , trans_b=0):
    A = np.ascontiguousarray(A , dtype=np.float32)
    B = np.ascontiguousarray(B , dtype=np.float32)
    if trans_a:
        output_row = A.shape[1]
        common = A.shape[0]
    else:
        output_row = A.shape[0]
        common = A.shape[1]
    if trans_b:
        output_col = B.shape[0]
    else:
        output_col = B.shape[1]

    C = np.empty((output_row , output_col) , dtype=np.float32)

    lib.matmul(A.ctypes.data_as(c_float_p) ,
               B.ctypes.data_as(c_float_p) ,
               C.ctypes.data_as(c_float_p) ,
               ctypes.c_int(output_row) ,
               ctypes.c_int(common) ,
               ctypes.c_int(output_col) ,
               ctypes.c_int(trans_a) ,
               ctypes.c_int(trans_b))

    return C

lib.matmul_bias.argtypes = [
    c_float_p, c_float_p, c_float_p, c_float_p,  # A, B, bias, C
    ctypes.c_int, ctypes.c_int, ctypes.c_int # M, K, N
]
lib.matmul_bias.restype = None

def matmul_bias(A, B, bias):
    A = np.ascontiguousarray(A, dtype=np.float32)
    B = np.ascontiguousarray(B, dtype=np.float32)
    bias = np.ascontiguousarray(bias, dtype=np.float32)

    M, K = A.shape
    N = B.shape[1]
    C = np.empty((M, N), dtype=np.float32)

    lib.matmul_bias(
        A.ctypes.data_as(c_float_p),
        B.ctypes.data_as(c_float_p),
        bias.ctypes.data_as(c_float_p),
        C.ctypes.data_as(c_float_p),
        ctypes.c_int(M), ctypes.c_int(K), ctypes.c_int(N)
    )

    return C

lib.add_bias.argtypes = [
    c_float_p , c_float_p , ctypes.c_int , ctypes.c_int
]
lib.add_bias.restype = None

def bias(A , bias):
    A = np.ascontiguousarray(A, dtype=np.float32)
    bias = np.ascontiguousarray(bias, dtype=np.float32)

    M, K = A.shape
    lib.add_bias(
        bias.ctypes.data_as(c_float_p),
        A.ctypes.data_as(c_float_p),
        ctypes.c_int(M),
        ctypes.c_int(K)
    )

    return A

lib.add_bias_row.argtypes = [
    c_float_p , c_float_p , ctypes.c_int , ctypes.c_int
]
lib.add_bias_row.restype = None

def bias_row(A , bias):
    A = np.ascontiguousarray(A, dtype=np.float32)
    bias = np.ascontiguousarray(bias, dtype=np.float32)

    M, K = A.shape
    lib.add_bias_row(
        bias.ctypes.data_as(c_float_p),
        A.ctypes.data_as(c_float_p),
        ctypes.c_int(M),
        ctypes.c_int(K)
    )

    return A

lib.im2col.argtypes = [
    c_float_p, c_float_p,  # input, col
    ctypes.c_int, ctypes.c_int, ctypes.c_int,  # C, H, W
    ctypes.c_int, ctypes.c_int,  # KH, KW
    ctypes.c_int, ctypes.c_int,  # stride, pad
    ctypes.c_int, ctypes.c_int   # H_out, W_out
]
lib.im2col.restype = None

def im2col(input, KH, KW, stride=1, pad=0):
    input = np.ascontiguousarray(input, dtype=np.float32)
    C, H, W = input.shape
    H_out = (H + 2 * pad - KH) // stride + 1
    W_out = (W + 2 * pad - KW) // stride + 1
    col = np.empty((H_out * W_out, C * KH * KW), dtype=np.float32)

    lib.im2col(
        input.ctypes.data_as(c_float_p),
        col.ctypes.data_as(c_float_p),
        ctypes.c_int(C), ctypes.c_int(H), ctypes.c_int(W),
        ctypes.c_int(KH), ctypes.c_int(KW),
        ctypes.c_int(stride), ctypes.c_int(pad),
        ctypes.c_int(H_out), ctypes.c_int(W_out)
    )
    return col

lib.conv2d_forward.argtypes = [
    c_float_p, c_float_p, c_float_p, c_float_p, c_float_p,  # input, weights, bias, output, col_buf
    ctypes.c_int, ctypes.c_int, ctypes.c_int,  # C, H, W
    ctypes.c_int,  # F
    ctypes.c_int, ctypes.c_int,  # KH, KW
    ctypes.c_int, ctypes.c_int   # stride, pad
]
lib.conv2d_forward.restype = None

def conv2d(input, weights, bias, stride=1, pad=0):
    input = np.ascontiguousarray(input, dtype=np.float32)
    weights = np.ascontiguousarray(weights, dtype=np.float32)
    bias = np.ascontiguousarray(bias, dtype=np.float32)

    C, H, W = input.shape
    F, _, KH, KW = weights.shape
    H_out = (H + 2 * pad - KH) // stride + 1
    W_out = (W + 2 * pad - KW) // stride + 1

    weights_flat = weights.reshape(F, C * KH * KW)
    output = np.empty((F, H_out * W_out), dtype=np.float32)
    col_buf = np.empty((H_out * W_out, C * KH * KW), dtype=np.float32)

    lib.conv2d_forward(
        input.ctypes.data_as(c_float_p),
        weights_flat.ctypes.data_as(c_float_p),
        bias.ctypes.data_as(c_float_p),
        output.ctypes.data_as(c_float_p),
        col_buf.ctypes.data_as(c_float_p),
        ctypes.c_int(C), ctypes.c_int(H), ctypes.c_int(W),
        ctypes.c_int(F),
        ctypes.c_int(KH), ctypes.c_int(KW),
        ctypes.c_int(stride), ctypes.c_int(pad)
    )
    return output.reshape(F, H_out, W_out)

lib.conv2d_forward_batch.argtypes = [
    c_float_p, c_float_p, c_float_p, c_float_p, c_float_p,  # input, weights, bias, output, col_buf
    ctypes.c_int,  # N
    ctypes.c_int, ctypes.c_int, ctypes.c_int,  # C, H, W
    ctypes.c_int,  # F
    ctypes.c_int, ctypes.c_int,  # KH, KW
    ctypes.c_int, ctypes.c_int   # stride, pad
]
lib.conv2d_forward_batch.restype = None

def conv2d_batch(input, weights, bias, stride=1, pad=0):
    input = np.ascontiguousarray(input, dtype=np.float32)
    weights = np.ascontiguousarray(weights, dtype=np.float32)
    bias = np.ascontiguousarray(bias, dtype=np.float32)

    N, C, H, W = input.shape
    F, _, KH, KW = weights.shape
    H_out = (H + 2 * pad - KH) // stride + 1
    W_out = (W + 2 * pad - KW) // stride + 1

    weights_flat = weights.reshape(F, C * KH * KW)
    output = np.empty((N, F, H_out * W_out), dtype=np.float32)
    col_buf = np.empty((H_out * W_out, C * KH * KW), dtype=np.float32)

    lib.conv2d_forward_batch(
        input.ctypes.data_as(c_float_p),
        weights_flat.ctypes.data_as(c_float_p),
        bias.ctypes.data_as(c_float_p),
        output.ctypes.data_as(c_float_p),
        col_buf.ctypes.data_as(c_float_p),
        ctypes.c_int(N),
        ctypes.c_int(C), ctypes.c_int(H), ctypes.c_int(W),
        ctypes.c_int(F),
        ctypes.c_int(KH), ctypes.c_int(KW),
        ctypes.c_int(stride), ctypes.c_int(pad)
    )
    return output.reshape(N, F, H_out, W_out)

lib.relu_forward.argtypes = [
    c_float_p, c_float_p, c_int_p,  # input, output, mask (int*)
    ctypes.c_int  # size
]
lib.relu_forward.restype = None

def relu_forward(input):
    input = np.ascontiguousarray(input, dtype=np.float32)
    output = np.empty(input.shape, dtype=np.float32)
    mask = np.empty(input.shape, dtype=np.int32)
    size = reduce(operator.mul, input.shape)

    lib.relu_forward(
        input.ctypes.data_as(c_float_p),
        output.ctypes.data_as(c_float_p),
        mask.ctypes.data_as(c_int_p),
        ctypes.c_int(size)
    )
    return output, mask

lib.leaky_relu_forward.argtypes = [
    c_float_p, c_float_p, c_float_p,  # input, output, mask (all float*)
    ctypes.c_int,  # size
    ctypes.c_float  # alpha
]
lib.leaky_relu_forward.restype = None

def leaky_relu_forward(input, alpha=0.01):
    input = np.ascontiguousarray(input, dtype=np.float32)
    output = np.empty(input.shape, dtype=np.float32)
    mask = np.empty(input.shape, dtype=np.float32)
    size = reduce(operator.mul, input.shape)

    lib.leaky_relu_forward(
        input.ctypes.data_as(c_float_p),
        output.ctypes.data_as(c_float_p),
        mask.ctypes.data_as(c_float_p),
        ctypes.c_int(size),
        ctypes.c_float(alpha)
    )
    return output, mask

lib.maxpool2d_forward.argtypes = [
    c_float_p, c_float_p, c_int_p,  # input, output, indices
    ctypes.c_int, ctypes.c_int, ctypes.c_int,  # C, H_out, W_out
    ctypes.c_int, ctypes.c_int,  # PH, PW
    ctypes.c_int  # stride
]
lib.maxpool2d_forward.restype = None

def maxpool2d_forward(input, PH, PW, stride):
    input = np.ascontiguousarray(input, dtype=np.float32)
    C, H_in, W_in = input.shape
    H_out = (H_in - PH) // stride + 1
    W_out = (W_in - PW) // stride + 1

    output = np.empty((C, H_out, W_out), dtype=np.float32)
    indices = np.empty((C, H_out, W_out), dtype=np.int32)

    lib.maxpool2d_forward(
        input.ctypes.data_as(c_float_p),
        output.ctypes.data_as(c_float_p),
        indices.ctypes.data_as(c_int_p),
        ctypes.c_int(C), ctypes.c_int(H_out), ctypes.c_int(W_out),
        ctypes.c_int(PH), ctypes.c_int(PW),
        ctypes.c_int(stride)
    )
    return output, indices
