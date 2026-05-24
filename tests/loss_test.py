import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python')))
import numpy as np
from loss import safe_softmax, cross_entropy_loss, softmax_cross_entropy_forward_backward

RTOL = 1e-5
ATOL = 1e-6

def check(name, expected, actual):
    if np.isscalar(expected):
        expected = np.array(expected)
        actual = np.array(actual)
    max_diff = np.max(np.abs(expected - actual))
    passed = np.allclose(expected, actual, rtol=RTOL, atol=ATOL)
    status = "PASSED" if passed else "FAILED"
    shape_str = f"shape={actual.shape}" if hasattr(actual, 'shape') else ""
    print(f"[{status}] {name:50s} | max_diff={max_diff:.2e} {shape_str}")
    if not passed:
        print(f"         expected={expected}, actual={actual}")
    return passed

all_passed = True

# ===================== softmax tests =====================

# --- Test 1: softmax basic ---
x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
out = safe_softmax(x)
expected = np.exp(x) / np.sum(np.exp(x))
all_passed &= check("softmax basic [1,3]", expected, out)

# --- Test 2: softmax sums to 1 ---
x = np.random.randn(4, 10).astype(np.float32)
out = safe_softmax(x)
all_passed &= check("softmax rows sum to 1", np.ones(4), out.sum(axis=-1))

# --- Test 3: softmax numerical stability (large values) ---
x = np.array([[1000.0, 1001.0, 1002.0]], dtype=np.float64)
out = safe_softmax(x)
expected = np.exp(np.array([0, 1, 2], dtype=np.float64))
expected = expected / expected.sum()
all_passed &= check("softmax large values (stability)", expected, out)

# --- Test 4: softmax batch ---
x = np.random.randn(8, 5).astype(np.float32)
out = safe_softmax(x)
all_passed &= check("softmax batch sums to 1", np.ones(8), out.sum(axis=-1).astype(np.float32))
assert np.all(out > 0), "softmax outputs should all be positive"
print(f"[PASSED] {'softmax all positive':50s} |")

# ===================== cross entropy tests =====================

# --- Test 5: CE perfect prediction ---
y_true = np.array([[0, 0, 1]], dtype=np.float32)
y_pred = np.array([[0.01, 0.01, 0.98]], dtype=np.float32)
loss = cross_entropy_loss(y_true, y_pred)
expected = -np.log(0.98)
all_passed &= check("CE near-perfect prediction", expected, loss)

# --- Test 6: CE uniform prediction (high loss) ---
y_true = np.array([[1, 0, 0]], dtype=np.float32)
y_pred = np.array([[1/3, 1/3, 1/3]], dtype=np.float32)
loss = cross_entropy_loss(y_true, y_pred)
expected = -np.log(1/3)
all_passed &= check("CE uniform prediction", expected, loss)

# --- Test 7: CE batch ---
y_true = np.eye(3, dtype=np.float32)  # 3 samples, 3 classes
y_pred = safe_softmax(np.random.randn(3, 3).astype(np.float32))
loss = cross_entropy_loss(y_true, y_pred)
expected = -np.mean([np.log(y_pred[i, i]) for i in range(3)])
all_passed &= check("CE batch [3,3]", expected, loss)

# ===================== combined forward+backward tests =====================

# --- Test 8: gradient numerical check ---
np.random.seed(42)
logits = np.random.randn(4, 5).astype(np.float64)  # float64 for numerical grad precision
y_true = np.zeros((4, 5), dtype=np.float64)
y_true[np.arange(4), np.random.randint(0, 5, 4)] = 1.0
N = logits.shape[0]

loss, grad = softmax_cross_entropy_forward_backward(logits, y_true, N)

# numerical gradient
eps = 1e-5
num_grad = np.zeros_like(logits)
for i in range(logits.shape[0]):
    for j in range(logits.shape[1]):
        logits_plus = logits.copy()
        logits_plus[i, j] += eps
        loss_plus, _ = softmax_cross_entropy_forward_backward(logits_plus, y_true, N)
        logits_minus = logits.copy()
        logits_minus[i, j] -= eps
        loss_minus, _ = softmax_cross_entropy_forward_backward(logits_minus, y_true, N)
        num_grad[i, j] = (loss_plus - loss_minus) / (2 * eps)

all_passed &= check("grad check: analytic vs numerical [4,5]", num_grad, grad)

# --- Test 9: gradient shape ---
logits = np.random.randn(8, 10).astype(np.float32)
y_true = np.zeros((8, 10), dtype=np.float32)
y_true[np.arange(8), np.random.randint(0, 10, 8)] = 1.0
loss, grad = softmax_cross_entropy_forward_backward(logits, y_true, 8)
all_passed &= check("grad shape matches logits", np.zeros((8, 10), dtype=np.float32), np.zeros_like(grad))
assert grad.shape == logits.shape, f"Shape mismatch: {grad.shape} vs {logits.shape}"
print(f"[PASSED] {'grad shape == logits shape':50s} | {grad.shape}")

# --- Test 10: gradient sums to ~0 per row ---
# softmax grad (p - y) sums to (1 - 1) = 0 per row before /N
row_sums = grad.sum(axis=-1)
all_passed &= check("grad rows sum to ~0", np.zeros(8), row_sums)

# --- Test 11: loss is non-negative ---
assert loss >= 0, f"Loss should be non-negative, got {loss}"
print(f"[PASSED] {'loss is non-negative':50s} | loss={loss:.6f}")

print(f"\n{'='*60}")
print(f"{'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
