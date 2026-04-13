"""Lightweight PGO profiling script for maturin --pgo builds.

Imports the Rust extension directly (forust.forust) to avoid pulling in pandas,
which has numpy ABI compatibility issues on Python 3.9/3.10 in manylinux containers.
Only depends on numpy (always available in maturin's temp PGO venv).
"""

import sys
import os

# Remove the source directory from sys.path so Python imports the installed
# wheel from site-packages, not the local forust/ source directory.
source_dir = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p) != source_dir]

import numpy as np

# Import the Rust extension directly — bypasses __init__.py (which imports pandas)
from forust.forust import GradientBooster

rows, cols = 5000, 20
rng = np.random.default_rng(42)
X_flat = np.ascontiguousarray(rng.standard_normal((rows, cols)).ravel(order="F"))
y = rng.integers(0, 2, rows).astype(np.float64)
sample_weight = np.ones(rows, dtype=np.float64)

# Train — exercises histogram construction, split finding, index pivoting
booster = GradientBooster(iterations=50, learning_rate=0.3, max_depth=5)
booster.fit(X_flat, rows, cols, y, sample_weight)

booster2 = GradientBooster(iterations=50, learning_rate=0.3, max_depth=5, subsample=0.5)
booster2.fit(X_flat, rows, cols, y, sample_weight)

booster3 = GradientBooster(iterations=50, learning_rate=0.3, max_depth=5, colsample_bytree=0.5)
booster3.fit(X_flat, rows, cols, y, sample_weight)

# Predict — single-threaded and parallel paths
booster.predict(X_flat, rows, cols, parallel=False)
booster.predict(X_flat, rows, cols, parallel=True)

# Contributions — exercises the tree-walk contribution paths
booster.predict_contributions(X_flat, rows, cols, method="Weight")
booster.predict_contributions(X_flat, rows, cols, method="Average")
booster.predict_contributions(X_flat, rows, cols, method="Shapley")

print("PGO profiling complete")

print("PGO profiling complete")
