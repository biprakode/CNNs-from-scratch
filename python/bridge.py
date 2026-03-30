import ctypes, os

import numpy as np

lib = ctypes.cdll.LoadLibrary("/run/media/biprarshi/COMMON/files/AI/CNNs_from_scratch/build/libcnn.so")

c_float_p = ctypes.POINTER(ctypes.c_float)

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
    ctypes.c_int, ctypes.c_int, ctypes.c_int       # M, K, N
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