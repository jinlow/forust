import pandas as pd
import numpy as np
from forust import  percentiles_nunique

df = pd.read_csv("../resources/titanic.csv").sample(1_000_000, replace=True, random_state=0)
percentiles = np.array([0.1, 0.2, 0.5, 0.78])
p, n = percentiles_nunique(df["fare"].to_numpy(), np.ones(df.shape[0]), percentiles)

p

df["fare"].nunique()
np.percentile(df["fare"], percentiles*100)


import pandas as pd
import numpy as np
from forust import GradientBooster

df = pd.read_csv("../resources/titanic.csv").sample(100_000, replace=True, random_state=0)

X = df.select_dtypes("number").drop(columns="survived").fillna(0).reset_index(drop=True)
X_vec = X.to_numpy().ravel(order="F")
y = df["survived"].to_numpy().astype("float64")
mod = GradientBooster(
        iterations=100,
        learning_rate=0.3,
        max_depth=5,
        l2=1,
        min_leaf_weight=1,
        gamma=0,
        objective_type="LogLoss",
)
mod.fit(X_vec, y.shape[0], 5, y, np.ones(y.shape))
mod.predict(X_vec, y.shape[0], 5)[0:10]

from sklearn.ensemble import HistGradientBoostingClassifier

# hgb = HistGradientBoostingClassifier(max_iter=100)
# hgb.fit(X, y)
# hgb.predict_proba(X)[0:10]

from xgboost import XGBClassifier
xmod = XGBClassifier(n_estimators=100, 
    learning_rate=0.3,
    max_depth=5,
    reg_lambda=1,
    min_child_weight=1,
    gamma=0,
    objective="binary:logitraw",
    eval_metric="auc",
)
xmod.fit(X, y)
xmod.predict(X, output_margin=True)[0:10]
