import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python')))
import numpy as np
from bridge import matmul, matmul_bias, bias as add_bias

RTOL = 1e-4
ATOL = 1e-5

def check(name, expected, actual):
    max_diff = np.max(np.abs(expected - actual))
    passed = np.allclose(expected, actual, rtol=RTOL, atol=ATOL)
    status = "PASSED" if passed else "FAILED"
    print(f"[{status}] {name:40s} | max_diff={max_diff:.2e} | shape={actual.shape}")
    if not passed:
        worst = np.unravel_index(np.argmax(np.abs(expected - actual)), expected.shape)
        print(f"         worst at {worst}: expected={expected[worst]:.6e}, actual={actual[worst]:.6e}")
    return passed

all_passed = True

# --- Test 1: Basic matmul ---
A = np.random.randn(64, 128).astype(np.float32)
B = np.random.randn(128, 32).astype(np.float32)
all_passed &= check("Basic matmul [64,128] @ [128,32]", A @ B, matmul(A, B))

# --- Test 2: matmul_bias ---
bias = np.random.randn(32).astype(np.float32)
all_passed &= check("matmul_bias [64,128] @ [128,32] + [32]", A @ B + bias, matmul_bias(A, B, bias))

# --- Test 3: trans_a --- A stored as [K, M], result = A.T @ B
A_stored = np.random.randn(128, 64).astype(np.float32)
B = np.random.randn(128, 32).astype(np.float32)
all_passed &= check("trans_a [128,64].T @ [128,32]", A_stored.T @ B, matmul(A_stored, B, trans_a=1))

# --- Test 4: trans_b --- B stored as [N, K], result = A @ B.T
A = np.random.randn(64, 128).astype(np.float32)
B_stored = np.random.randn(32, 128).astype(np.float32)
all_passed &= check("trans_b [64,128] @ [32,128].T", A @ B_stored.T, matmul(A, B_stored, trans_b=1))

# --- Test 5: both transposed --- A stored [K,M], B stored [N,K], result = A.T @ B.T
A_stored = np.random.randn(128, 64).astype(np.float32)
B_stored = np.random.randn(32, 128).astype(np.float32)
all_passed &= check("trans_a+b [128,64].T @ [32,128].T", A_stored.T @ B_stored.T, matmul(A_stored, B_stored, trans_a=1, trans_b=1))

# --- Test 6: edge case M=1 (single row) ---
A = np.random.randn(1, 128).astype(np.float32)
B = np.random.randn(128, 32).astype(np.float32)
all_passed &= check("edge M=1 [1,128] @ [128,32]", A @ B, matmul(A, B))

# --- Test 7: edge case N=1 (single column) ---
A = np.random.randn(64, 128).astype(np.float32)
B = np.random.randn(128, 1).astype(np.float32)
all_passed &= check("edge N=1 [64,128] @ [128,1]", A @ B, matmul(A, B))

# --- Test 8: edge case K=1 (trivial inner dim) ---
A = np.random.randn(64, 1).astype(np.float32)
B = np.random.randn(1, 32).astype(np.float32)
all_passed &= check("edge K=1 [64,1] @ [1,32]", A @ B, matmul(A, B))

# --- Test 9: basic add_bias [4,8] + [8] ---
A = np.random.randn(4, 8).astype(np.float32)
b = np.random.randn(8).astype(np.float32)
expected = A + b
all_passed &= check("add_bias [4,8] + [8]", expected, add_bias(A.copy(), b))

# --- Test 10: add_bias single row M=1 ---
A = np.random.randn(1, 16).astype(np.float32)
b = np.random.randn(16).astype(np.float32)
expected = A + b
all_passed &= check("add_bias M=1 [1,16] + [16]", expected, add_bias(A.copy(), b))

# --- Test 11: add_bias single col K=1 ---
A = np.random.randn(8, 1).astype(np.float32)
b = np.random.randn(1).astype(np.float32)
expected = A + b
all_passed &= check("add_bias K=1 [8,1] + [1]", expected, add_bias(A.copy(), b))

# --- Test 12: add_bias large ---
A = np.random.randn(256, 512).astype(np.float32)
b = np.random.randn(512).astype(np.float32)
expected = A + b
all_passed &= check("add_bias large [256,512] + [512]", expected, add_bias(A.copy(), b))

print(f"\n{'='*60}")
print(f"{'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
