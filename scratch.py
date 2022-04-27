from typing import Tuple
import seaborn as sns
import xgboost as xgb
from pyforust.model import LogLoss
import numpy as np

ll = LogLoss()


def log_loss(predt: np.array, dtrain: xgb.DMatrix) -> Tuple[np.array, np.array]:
    y = dtrain.get_label()
    # predt = 1/(1+np.exp(-predt))
    # print(predt)
    g = ll.grad(y, predt)
    h = ll.hess(y, predt)
    return g, h


df = sns.load_dataset("titanic")
X = df.select_dtypes("number").drop(columns="survived")
y = df["survived"]

d = xgb.DMatrix(X, label=y)

# mod = xgb.train(
#     {
#         "seed": 123,
#         "objective": "binary:logitraw",
#         "eval_metric": "auc",
#     },
#     num_boost_round=15,
#     dtrain=d,
#     evals=[(d, "train")],
# )

# mod = xgb.train(
#     {
#         "seed": 123,
#         "eval_metric": "auc",
#     },
#     num_boost_round=15,
#     dtrain=d,
#     evals=[(d, "train")],
#     obj=log_loss,
# )


