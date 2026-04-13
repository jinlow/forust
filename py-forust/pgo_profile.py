"""Lightweight PGO profiling script for maturin --pgo builds.

Exercises all hot Rust code paths so the compiler collects useful branch/call data.
Pandas is lazily imported in forust, so this works even if pandas has issues.
"""

import os
import sys

# Remove the source directory from sys.path so Python imports the installed
# wheel from site-packages, not the local forust/ source directory.
source_dir = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p) != source_dir]

import numpy as np

from forust import GradientBooster

rng = np.random.default_rng(42)
X = rng.standard_normal((5000, 20))
y = rng.integers(0, 2, 5000).astype(np.float64)

# Train — exercises histogram construction, split finding, index pivoting
booster = GradientBooster(iterations=50, learning_rate=0.3, max_depth=5)
booster.fit(X, y)

booster2 = GradientBooster(iterations=50, learning_rate=0.3, max_depth=5, subsample=0.5)
booster2.fit(X, y)

booster3 = GradientBooster(
    iterations=50, learning_rate=0.3, max_depth=5, colsample_bytree=0.5
)
booster3.fit(X, y)

# Predict — single-threaded and parallel paths
booster.predict(X)
booster.predict(X, parallel=True)

# Contributions — exercises the tree-walk contribution paths
booster.predict_contributions(X, method="Weight")
booster.predict_contributions(X, method="Average")
booster.predict_contributions(X, method="Shapley")

print("PGO profiling complete")
