"""
Generate reference values from Python XGBoost for cross-validation
against our forust-ml SoftmaxMultiClass implementation.

Tests 3 things:
1. Softmax math (identical formula)
2. Gradient/Hessian values (identical formula)
3. Base score initialization (identical formula)
4. End-to-end training on same dataset → compare accuracy (not exact logits)
"""

import json
import numpy as np
import xgboost as xgb

print(f"XGBoost version: {xgb.__version__}")

# ─── 1. Softmax reference ───────────────────────────────────────────────
# XGBoost uses the same numerically-stable softmax as our implementation
def reference_softmax(scores, K):
    """Exactly matches xgboost/src/common/math.h Softmax"""
    n = len(scores) // K
    probs = []
    for i in range(n):
        row = scores[i*K:(i+1)*K]
        wmax = max(row)
        exps = [np.exp(s - wmax) for s in row]
        wsum = sum(exps)
        probs.extend([e / wsum for e in exps])
    return probs

softmax_inputs = [
    [1.0, 2.0, 3.0],                    # normal
    [1000.0, 1000.0, 1000.0],           # extreme equal
    [-1000.0, 0.0, 1000.0],             # extreme range
    [0.0, 0.0, 0.0],                    # zeros
    [1.0, 2.0, 3.0, -1.0, 0.0, 1.0],   # 2 rows
]

softmax_results = {}
for i, inp in enumerate(softmax_inputs):
    probs = reference_softmax(inp, 3)
    softmax_results[f"case_{i}"] = {
        "input": inp,
        "K": 3,
        "output": probs,
    }
    print(f"Softmax case {i}: sum(probs)={sum(probs[:3]):.15f}")

# ─── 2. Gradient/Hessian reference ──────────────────────────────────────
# Directly from multiclass_obj.cu GetGradient
def reference_grad_hess(y, yhat, w, K):
    K_EPS = 1e-16
    n = len(y)
    grad = [0.0] * (n * K)
    hess = [0.0] * (n * K)
    for i in range(n):
        row = yhat[i*K:(i+1)*K]
        wmax = max(row)
        wsum = sum(np.exp(s - wmax) for s in row)
        label = int(y[i])
        wt = w[i]
        for k in range(K):
            p = float(np.exp(row[k] - wmax) / wsum)
            idx = i * K + k
            hess[idx] = max(2.0 * p * (1.0 - p) * wt, K_EPS)
            p_adj = (p - 1.0) if k == label else p
            grad[idx] = p_adj * wt
    return grad, hess

gh_cases = [
    {
        "y": [0.0],
        "yhat": [1.0, 0.0, 0.0],
        "w": [1.0],
        "K": 3,
    },
    {
        "y": [1.0, 2.0],
        "yhat": [0.5, 0.5, 0.5, -1.0, 0.0, 2.0],
        "w": [1.0, 2.0],
        "K": 3,
    },
    {
        "y": [0.0, 1.0, 2.0],
        "yhat": [0.0]*9,
        "w": [1.0, 1.0, 1.0],
        "K": 3,
    },
]

grad_hess_results = {}
for i, case in enumerate(gh_cases):
    g, h = reference_grad_hess(case["y"], case["yhat"], case["w"], case["K"])
    grad_hess_results[f"case_{i}"] = {
        **case,
        "grad": g,
        "hess": h,
    }
    print(f"GradHess case {i}: grad[0]={g[0]:.10f}, hess[0]={h[0]:.10f}")

# ─── 3. Base score (InitEstimation) reference ───────────────────────────
def reference_calc_init(y, w, K):
    RT_EPS = 1e-15
    class_weights = [0.0] * K
    total = 0.0
    for yi, wi in zip(y, w):
        k = int(yi)
        if k < K:
            class_weights[k] += wi
        total += wi
    logs = [np.log(max(cw / total, RT_EPS)) for cw in class_weights]
    mean_log = sum(logs) / K
    return [l - mean_log for l in logs]

init_cases = [
    {"y": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0], "w": [1.0]*6, "K": 3},  # balanced
    {"y": [0.0, 0.0, 0.0, 1.0, 2.0, 2.0], "w": [1.0]*6, "K": 3},  # imbalanced
    {"y": [0.0, 0.0, 1.0, 1.0], "w": [1.0, 2.0, 1.0, 2.0], "K": 2},  # weighted K=2
]

init_results = {}
for i, case in enumerate(init_cases):
    base = reference_calc_init(case["y"], case["w"], case["K"])
    init_results[f"case_{i}"] = {
        **case,
        "base_scores": base,
    }
    print(f"CalcInit case {i}: base_scores={[f'{b:.10f}' for b in base]}")

# ─── 4. End-to-end training comparison ──────────────────────────────────
# Generate the SAME synthetic 3-class dataset used in our Rust tests
np.random.seed(42)
n_per_class = 100
n = n_per_class * 3
n_cols = 2

# Class 0 at (0,0), class 1 at (5,5), class 2 at (10,0) — same as Rust
X = np.zeros((n, n_cols))
y_train = np.zeros(n)
for i in range(n_per_class):
    offset = i * 0.02 - 1.0
    X[i] = [0.0 + offset, 0.0 + offset]
    y_train[i] = 0
for i in range(n_per_class):
    offset = i * 0.02 - 1.0
    X[n_per_class + i] = [5.0 + offset, 5.0 + offset]
    y_train[n_per_class + i] = 1
for i in range(n_per_class):
    offset = i * 0.02 - 1.0
    X[2*n_per_class + i] = [10.0 + offset, 0.0 + offset]
    y_train[2*n_per_class + i] = 2

# Train XGBoost with multi:softprob
params = {
    "objective": "multi:softprob",
    "num_class": 3,
    "max_depth": 4,
    "learning_rate": 0.3,
    "n_estimators": 50,
    "seed": 0,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "min_child_weight": 1.0,
    "reg_lambda": 1.0,  # L2 reg (forust default)
    "reg_alpha": 0.0,   # L1 reg
    "gamma": 0.0,
}

dtrain = xgb.DMatrix(X, label=y_train)
bst = xgb.train(
    {k: v for k, v in params.items() if k != "n_estimators"},
    dtrain,
    num_boost_round=params["n_estimators"],
)

# Get probabilities
probs_xgb = bst.predict(dtrain)  # shape (300, 3)
# Get raw logits
logits_xgb = bst.predict(dtrain, output_margin=True)

# Accuracy
preds_xgb = np.argmax(probs_xgb, axis=1)
accuracy_xgb = np.mean(preds_xgb == y_train)
print(f"\nXGBoost accuracy: {accuracy_xgb:.4f}")
print(f"XGBoost probs[0]: {probs_xgb[0]}")
print(f"XGBoost logits[0]: {logits_xgb[0]}")

# Verify probs sum to 1
prob_sums = probs_xgb.sum(axis=1)
print(f"XGBoost prob sums: min={prob_sums.min():.10f}, max={prob_sums.max():.10f}")

# ─── Save all reference data ────────────────────────────────────────────
reference = {
    "xgboost_version": xgb.__version__,
    "softmax": softmax_results,
    "grad_hess": grad_hess_results,
    "calc_init": init_results,
    "end_to_end": {
        "params": params,
        "n_samples": n,
        "n_features": n_cols,
        "accuracy": float(accuracy_xgb),
        "probs_first_5": probs_xgb[:5].tolist(),
        "logits_first_5": logits_xgb[:5].tolist(),
        "probs_last_5": probs_xgb[-5:].tolist(),
    },
    # Save the dataset in column-major (forust-ml format) for Rust test
    "dataset": {
        "X_col_major": [float(X[row, col]) for col in range(n_cols) for row in range(n)],
        "y": y_train.tolist(),
        "n_rows": n,
        "n_cols": n_cols,
    },
}

output_path = "tests/xgb_reference.json"
with open(output_path, "w") as f:
    json.dump(reference, f, indent=2)

print(f"\nReference data saved to {output_path}")
print(f"Keys: {list(reference.keys())}")
