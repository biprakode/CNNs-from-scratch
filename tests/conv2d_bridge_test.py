import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python')))
import numpy as np
from bridge import im2col, conv2d, conv2d_batch

RTOL = 1e-4
ATOL = 1e-5

def check(name, expected, actual):
    max_diff = np.max(np.abs(expected - actual))
    passed = np.allclose(expected, actual, rtol=RTOL, atol=ATOL)
    status = "PASSED" if passed else "FAILED"
    print(f"[{status}] {name:45s} | max_diff={max_diff:.2e} | shape={actual.shape}")
    if not passed:
        worst = np.unravel_index(np.argmax(np.abs(expected - actual)), expected.shape)
        print(f"         worst at {worst}: expected={expected[worst]:.6e}, actual={actual[worst]:.6e}")
    return passed

def numpy_im2col(input, KH, KW, stride=1, pad=0):
    C, H, W = input.shape
    if pad > 0:
        input = np.pad(input, ((0, 0), (pad, pad), (pad, pad)))
    H_pad, W_pad = input.shape[1], input.shape[2]
    H_out = (H + 2 * pad - KH) // stride + 1
    W_out = (W + 2 * pad - KW) // stride + 1
    col = np.empty((H_out * W_out, C * KH * KW), dtype=np.float32)
    for i in range(H_out):
        for j in range(W_out):
            patch = input[:, i*stride:i*stride+KH, j*stride:j*stride+KW]
            col[i * W_out + j] = patch.flatten()
    return col

def numpy_conv2d(input, weights, bias, stride=1, pad=0):
    col = numpy_im2col(input, weights.shape[2], weights.shape[3], stride, pad)
    F = weights.shape[0]
    W_flat = weights.reshape(F, -1)
    out = W_flat @ col.T + bias[:, None]
    H_out = (input.shape[1] + 2 * pad - weights.shape[2]) // stride + 1
    W_out = (input.shape[2] + 2 * pad - weights.shape[3]) // stride + 1
    return out.reshape(F, H_out, W_out)

all_passed = True

# ===================== im2col tests =====================

# --- Test 1: im2col basic 1 channel, 3x3 kernel, no pad ---
inp = np.random.randn(1, 5, 5).astype(np.float32)
all_passed &= check("im2col [1,5,5] k=3 s=1 p=0", numpy_im2col(inp, 3, 3), im2col(inp, 3, 3))

# --- Test 2: im2col 3 channels ---
inp = np.random.randn(3, 5, 5).astype(np.float32)
all_passed &= check("im2col [3,5,5] k=3 s=1 p=0", numpy_im2col(inp, 3, 3), im2col(inp, 3, 3))

# --- Test 3: im2col with padding ---
inp = np.random.randn(1, 5, 5).astype(np.float32)
all_passed &= check("im2col [1,5,5] k=3 s=1 p=1", numpy_im2col(inp, 3, 3, pad=1), im2col(inp, 3, 3, pad=1))

# --- Test 4: im2col with stride ---
inp = np.random.randn(1, 6, 6).astype(np.float32)
all_passed &= check("im2col [1,6,6] k=3 s=2 p=0", numpy_im2col(inp, 3, 3, stride=2), im2col(inp, 3, 3, stride=2))

# --- Test 5: im2col stride + padding ---
inp = np.random.randn(3, 8, 8).astype(np.float32)
all_passed &= check("im2col [3,8,8] k=3 s=2 p=1", numpy_im2col(inp, 3, 3, stride=2, pad=1), im2col(inp, 3, 3, stride=2, pad=1))

# --- Test 6: im2col 1x1 kernel ---
inp = np.random.randn(3, 4, 4).astype(np.float32)
all_passed &= check("im2col [3,4,4] k=1 s=1 p=0", numpy_im2col(inp, 1, 1), im2col(inp, 1, 1))

# --- Test 7: im2col non-square kernel ---
inp = np.random.randn(2, 6, 8).astype(np.float32)
all_passed &= check("im2col [2,6,8] kh=3 kw=5 s=1 p=0", numpy_im2col(inp, 3, 5), im2col(inp, 3, 5))

# ===================== conv2d tests =====================

# --- Test 8: conv2d basic ---
inp = np.random.randn(1, 5, 5).astype(np.float32)
w = np.random.randn(1, 1, 3, 3).astype(np.float32)
b = np.random.randn(1).astype(np.float32)
all_passed &= check("conv2d [1,5,5] F=1 k=3 s=1 p=0", numpy_conv2d(inp, w, b), conv2d(inp, w, b))

# --- Test 9: conv2d multi-channel input ---
inp = np.random.randn(3, 8, 8).astype(np.float32)
w = np.random.randn(4, 3, 3, 3).astype(np.float32)
b = np.random.randn(4).astype(np.float32)
all_passed &= check("conv2d [3,8,8] F=4 k=3 s=1 p=0", numpy_conv2d(inp, w, b), conv2d(inp, w, b))

# --- Test 10: conv2d with padding ---
inp = np.random.randn(3, 8, 8).astype(np.float32)
w = np.random.randn(4, 3, 3, 3).astype(np.float32)
b = np.random.randn(4).astype(np.float32)
all_passed &= check("conv2d [3,8,8] F=4 k=3 s=1 p=1", numpy_conv2d(inp, w, b, pad=1), conv2d(inp, w, b, pad=1))

# --- Test 11: conv2d with stride ---
inp = np.random.randn(3, 8, 8).astype(np.float32)
w = np.random.randn(2, 3, 3, 3).astype(np.float32)
b = np.random.randn(2).astype(np.float32)
all_passed &= check("conv2d [3,8,8] F=2 k=3 s=2 p=0", numpy_conv2d(inp, w, b, stride=2), conv2d(inp, w, b, stride=2))

# --- Test 12: conv2d stride + padding ---
inp = np.random.randn(3, 16, 16).astype(np.float32)
w = np.random.randn(8, 3, 3, 3).astype(np.float32)
b = np.random.randn(8).astype(np.float32)
all_passed &= check("conv2d [3,16,16] F=8 k=3 s=2 p=1", numpy_conv2d(inp, w, b, stride=2, pad=1), conv2d(inp, w, b, stride=2, pad=1))

# --- Test 13: conv2d 1x1 kernel ---
inp = np.random.randn(3, 4, 4).astype(np.float32)
w = np.random.randn(8, 3, 1, 1).astype(np.float32)
b = np.random.randn(8).astype(np.float32)
all_passed &= check("conv2d [3,4,4] F=8 k=1 s=1 p=0", numpy_conv2d(inp, w, b), conv2d(inp, w, b))

# ===================== conv2d_batch tests =====================

def numpy_conv2d_batch(input, weights, bias, stride=1, pad=0):
    return np.stack([numpy_conv2d(input[i], weights, bias, stride, pad) for i in range(input.shape[0])])

# --- Test 14: conv2d_batch basic N=2 ---
inp = np.random.randn(2, 1, 5, 5).astype(np.float32)
w = np.random.randn(1, 1, 3, 3).astype(np.float32)
b = np.random.randn(1).astype(np.float32)
all_passed &= check("conv2d_batch N=2 [1,5,5] F=1 k=3 s=1 p=0", numpy_conv2d_batch(inp, w, b), conv2d_batch(inp, w, b))

# --- Test 15: conv2d_batch N=4 multi-channel ---
inp = np.random.randn(4, 3, 8, 8).astype(np.float32)
w = np.random.randn(4, 3, 3, 3).astype(np.float32)
b = np.random.randn(4).astype(np.float32)
all_passed &= check("conv2d_batch N=4 [3,8,8] F=4 k=3 s=1 p=0", numpy_conv2d_batch(inp, w, b), conv2d_batch(inp, w, b))

# --- Test 16: conv2d_batch with padding ---
inp = np.random.randn(3, 3, 8, 8).astype(np.float32)
w = np.random.randn(4, 3, 3, 3).astype(np.float32)
b = np.random.randn(4).astype(np.float32)
all_passed &= check("conv2d_batch N=3 [3,8,8] F=4 k=3 s=1 p=1", numpy_conv2d_batch(inp, w, b, pad=1), conv2d_batch(inp, w, b, pad=1))

# --- Test 17: conv2d_batch with stride ---
inp = np.random.randn(2, 3, 8, 8).astype(np.float32)
w = np.random.randn(2, 3, 3, 3).astype(np.float32)
b = np.random.randn(2).astype(np.float32)
all_passed &= check("conv2d_batch N=2 [3,8,8] F=2 k=3 s=2 p=0", numpy_conv2d_batch(inp, w, b, stride=2), conv2d_batch(inp, w, b, stride=2))

# --- Test 18: conv2d_batch stride + padding ---
inp = np.random.randn(4, 3, 16, 16).astype(np.float32)
w = np.random.randn(8, 3, 3, 3).astype(np.float32)
b = np.random.randn(8).astype(np.float32)
all_passed &= check("conv2d_batch N=4 [3,16,16] F=8 k=3 s=2 p=1", numpy_conv2d_batch(inp, w, b, stride=2, pad=1), conv2d_batch(inp, w, b, stride=2, pad=1))

# --- Test 19: conv2d_batch N=1 (single image) ---
inp = np.random.randn(1, 3, 8, 8).astype(np.float32)
w = np.random.randn(4, 3, 3, 3).astype(np.float32)
b = np.random.randn(4).astype(np.float32)
all_passed &= check("conv2d_batch N=1 [3,8,8] F=4 k=3 s=1 p=0", numpy_conv2d_batch(inp, w, b), conv2d_batch(inp, w, b))

print(f"\n{'='*60}")
print(f"{'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
