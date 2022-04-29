import seaborn as sns
import xgboost as xgb
from pyforust.model import LogLoss, Tree
import numpy as np
from typing import Tuple

t = Tree(max_depth=1)
ll = LogLoss()

df = sns.load_dataset("titanic")
X = df.select_dtypes("number").drop(columns="survived").fillna(0)
y = df["survived"]

init_preds = np.repeat(0.5, y.shape)
g = ll.grad(y, init_preds)
h = ll.hess(y, init_preds)
t.fit(X.to_numpy(), grad=g, hess=h)
print(len(t.nodes))
print(t.nodes[0])
print(t.nodes[1].weight_value)
print(t.nodes[2].weight_value)

def log_loss(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
    y = dtrain.get_label()
    # predt = 1/(1+np.exp(-predt))
    # print(predt)
    print(len(predt))
    init_preds = np.repeat(0.5, y.shape)
    g = ll.grad(y, init_preds)
    h = ll.hess(y, init_preds)
    # print(h.sum())
    # print(h[0:10])
    # print(np.unique(y))
    return g, h

d = xgb.DMatrix(X, label=y)
mod = xgb.train(
    {
        "seed": 123,
        # "objective": "binary:logitraw",
        "eval_metric": "auc",
        "lambda": 1,
        "gamma": 5,
        "max_depth": 1,
    },
    obj=log_loss,
    num_boost_round=1,
    dtrain=d,
    evals=[(d, "train")],
)
mod.get_score(importance_type="gain")
y[X["pclass"].lt(3)].mean(), y[X["pclass"].ge(3)].mean()
print(mod.get_dump(with_stats=True)[0])

# from typing import Tuple
# import seaborn as sns
# import xgboost as xgb
# from pyforust.model import LogLoss, TreeNode
# import numpy as np

# ll = LogLoss()


# def log_loss(predt: np.array, dtrain: xgb.DMatrix) -> Tuple[np.array, np.array]:
#     y = dtrain.get_label()
#     # predt = 1/(1+np.exp(-predt))
#     # print(predt)
#     g = ll.grad(y, predt)
#     h = ll.hess(y, predt)
#     return g, h


# df = sns.load_dataset("titanic")
# X = df.select_dtypes("number").drop(columns="survived").fillna(0)
# y = df["survived"]


# ## TreeNode
# ll = LogLoss
# tn = TreeNode(lambda_val=1, gamma_val=1, min_child_weight=1)
# init_preds = np.repeat(0.5, y.shape)
# g = ll.grad(y, init_preds)
# h = ll.hess(y, init_preds)
# tn.fit(X.to_numpy(), g, h)
# tn.split_
# fi, v, gv  = tn.split_
# f = X.columns[fi]
# print(f)
# y[X[f].lt(v)].mean(), y[X[f].ge(v)].mean()

# # In the actual stats returned for the model
# # gamma is not used, I think it's
# # only actually used in the evaluation
# # of the split.
# tn.split_gain(g, h, X["pclass"].lt(3).to_numpy(), X["pclass"].ge(3).to_numpy())
# # tn.split_gain(g, h, X["pclass"].lt(10000).to_numpy(), X["pclass"].ge(100000).to_numpy())
# # print((g.sum()**2)/(h.sum() + 1))

# # from pyforust.test_model import Node

# # n = Node(
# #     X.to_numpy(),
# #     g,
# #     h,
# #     np.array(np.arange(X.shape[0])),
# # )

# def log_loss(predt: np.array, dtrain: xgb.DMatrix) -> Tuple[np.array, np.array]:
#     y = dtrain.get_label()
#     # predt = 1/(1+np.exp(-predt))
#     # print(predt)
#     print(len(predt))
#     init_preds = np.repeat(0.5, y.shape)
#     g = ll.grad(y, init_preds)
#     h = ll.hess(y, init_preds)
#     # print(h.sum())
#     # print(h[0:10])
#     # print(np.unique(y))
#     return g, h

# d = xgb.DMatrix(X, label=y)
# mod = xgb.train(
#     {
#         "seed": 123,
#         # "objective": "binary:logitraw",
#         "eval_metric": "auc",
#         "lambda": 1,
#         "gamma": 5,
#         "max_depth": 1,
#     },
#     obj=log_loss,
#     num_boost_round=1,
#     dtrain=d,
#     evals=[(d, "train")],
# )
# mod.get_score(importance_type="gain")
# y[X["pclass"].lt(3)].mean(), y[X["pclass"].ge(3)].mean()
# print(mod.get_dump(with_stats=True)[0])
# # mod = xgb.train(
# #     {
# #         "seed": 123,
# #         "eval_metric": "auc",
# #     },
# #     num_boost_round=15,
# #     dtrain=d,
# #     evals=[(d, "train")],
# #     obj=log_loss,
# # )


