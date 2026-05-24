import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python')))
import numpy as np
from bridge import relu_forward, leaky_relu_forward, maxpool2d_forward

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

all_passed = True

# ===================== relu tests =====================

# --- Test 1: relu basic ---
x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
out, mask = relu_forward(x)
all_passed &= check("relu basic", np.maximum(x, 0), out)
all_passed &= check("relu mask", np.array([0, 0, 0, 1, 1], dtype=np.int32), mask)

# --- Test 2: relu 2D ---
x = np.random.randn(4, 8).astype(np.float32)
out, mask = relu_forward(x)
all_passed &= check("relu 2D [4,8]", np.maximum(x, 0), out)
all_passed &= check("relu 2D mask", (x > 0).astype(np.int32), mask)

# --- Test 3: relu 3D (like conv output) ---
x = np.random.randn(8, 6, 6).astype(np.float32)
out, mask = relu_forward(x)
all_passed &= check("relu 3D [8,6,6]", np.maximum(x, 0), out)
all_passed &= check("relu 3D mask", (x > 0).astype(np.int32), mask)

# --- Test 4: relu all negative ---
x = np.array([-5.0, -3.0, -1.0, -0.5], dtype=np.float32)
out, mask = relu_forward(x)
all_passed &= check("relu all negative", np.zeros(4, dtype=np.float32), out)

# --- Test 5: relu all positive ---
x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
out, mask = relu_forward(x)
all_passed &= check("relu all positive", x, out)

# ===================== leaky relu tests =====================

# --- Test 6: leaky relu basic ---
x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
alpha = 0.1
out, mask = leaky_relu_forward(x, alpha=alpha)
expected = np.where(x > 0, x, x * alpha).astype(np.float32)
all_passed &= check("leaky_relu basic a=0.1", expected, out)
expected_mask = np.where(x > 0, 1.0, alpha).astype(np.float32)
all_passed &= check("leaky_relu mask a=0.1", expected_mask, mask)

# --- Test 7: leaky relu 2D ---
x = np.random.randn(4, 8).astype(np.float32)
alpha = 0.01
out, mask = leaky_relu_forward(x, alpha=alpha)
expected = np.where(x > 0, x, x * alpha).astype(np.float32)
all_passed &= check("leaky_relu 2D [4,8] a=0.01", expected, out)

# --- Test 8: leaky relu 3D ---
x = np.random.randn(8, 6, 6).astype(np.float32)
alpha = 0.2
out, mask = leaky_relu_forward(x, alpha=alpha)
expected = np.where(x > 0, x, x * alpha).astype(np.float32)
all_passed &= check("leaky_relu 3D [8,6,6] a=0.2", expected, out)
expected_mask = np.where(x > 0, 1.0, alpha).astype(np.float32)
all_passed &= check("leaky_relu 3D mask a=0.2", expected_mask, mask)

# --- Test 9: leaky relu alpha=0 (should equal relu) ---
x = np.random.randn(16).astype(np.float32)
out, _ = leaky_relu_forward(x, alpha=0.0)
all_passed &= check("leaky_relu a=0 == relu", np.maximum(x, 0), out)

# --- Test 10: leaky relu large batch ---
x = np.random.randn(32, 16, 16).astype(np.float32)
alpha = 0.01
out, mask = leaky_relu_forward(x, alpha=alpha)
expected = np.where(x > 0, x, x * alpha).astype(np.float32)
all_passed &= check("leaky_relu large [32,16,16] a=0.01", expected, out)

# ===================== maxpool2d tests =====================

def numpy_maxpool2d(input, PH, PW, stride):
    C, H_in, W_in = input.shape
    H_out = (H_in - PH) // stride + 1
    W_out = (W_in - PW) // stride + 1
    output = np.empty((C, H_out, W_out), dtype=np.float32)
    indices = np.empty((C, H_out, W_out), dtype=np.int32)
    for c in range(C):
        for oh in range(H_out):
            for ow in range(W_out):
                patch = input[c, oh*stride:oh*stride+PH, ow*stride:ow*stride+PW]
                local_idx = np.argmax(patch)
                ph, pw = np.unravel_index(local_idx, (PH, PW))
                ih, iw = oh * stride + ph, ow * stride + pw
                output[c, oh, ow] = input[c, ih, iw]
                indices[c, oh, ow] = c * H_in * W_in + ih * W_in + iw
    return output, indices

# --- Test 11: maxpool basic 1 channel 2x2 s=2 ---
x = np.random.randn(1, 4, 4).astype(np.float32)
out, idx = maxpool2d_forward(x, 2, 2, 2)
exp_out, exp_idx = numpy_maxpool2d(x, 2, 2, 2)
all_passed &= check("maxpool [1,4,4] p=2 s=2 output", exp_out, out)
all_passed &= check("maxpool [1,4,4] p=2 s=2 indices", exp_idx, idx)

# --- Test 12: maxpool multi-channel ---
x = np.random.randn(3, 4, 4).astype(np.float32)
out, idx = maxpool2d_forward(x, 2, 2, 2)
exp_out, exp_idx = numpy_maxpool2d(x, 2, 2, 2)
all_passed &= check("maxpool [3,4,4] p=2 s=2 output", exp_out, out)
all_passed &= check("maxpool [3,4,4] p=2 s=2 indices", exp_idx, idx)

# --- Test 13: maxpool stride=1 (overlapping) ---
x = np.random.randn(2, 4, 4).astype(np.float32)
out, idx = maxpool2d_forward(x, 2, 2, 1)
exp_out, exp_idx = numpy_maxpool2d(x, 2, 2, 1)
all_passed &= check("maxpool [2,4,4] p=2 s=1 output", exp_out, out)
all_passed &= check("maxpool [2,4,4] p=2 s=1 indices", exp_idx, idx)

# --- Test 14: maxpool 3x3 pool ---
x = np.random.randn(1, 6, 6).astype(np.float32)
out, idx = maxpool2d_forward(x, 3, 3, 3)
exp_out, exp_idx = numpy_maxpool2d(x, 3, 3, 3)
all_passed &= check("maxpool [1,6,6] p=3 s=3 output", exp_out, out)
all_passed &= check("maxpool [1,6,6] p=3 s=3 indices", exp_idx, idx)

# --- Test 15: maxpool larger ---
x = np.random.randn(8, 16, 16).astype(np.float32)
out, idx = maxpool2d_forward(x, 2, 2, 2)
exp_out, exp_idx = numpy_maxpool2d(x, 2, 2, 2)
all_passed &= check("maxpool [8,16,16] p=2 s=2 output", exp_out, out)
all_passed &= check("maxpool [8,16,16] p=2 s=2 indices", exp_idx, idx)

# --- Test 16: maxpool non-square pool ---
x = np.random.randn(2, 6, 8).astype(np.float32)
out, idx = maxpool2d_forward(x, 2, 4, 2)
exp_out, exp_idx = numpy_maxpool2d(x, 2, 4, 2)
all_passed &= check("maxpool [2,6,8] ph=2 pw=4 s=2 output", exp_out, out)
all_passed &= check("maxpool [2,6,8] ph=2 pw=4 s=2 indices", exp_idx, idx)

print(f"\n{'='*60}")
print(f"{'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
