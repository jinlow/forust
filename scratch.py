import seaborn as sns
import xgboost as xgb
from forust.model import LogLoss, Tree, XGBoost, weight
import numpy as np
from typing import Tuple

# t = Tree(max_depth=2)
t = Tree(
    l2=1.0,
    gamma=3.0,
    min_leaf_weight=1.0,
    learning_rate=0.3,
    max_depth=5,
)
ll = LogLoss()

df = sns.load_dataset("titanic")
# Need to add NA support
X = df.select_dtypes("number").drop(columns="survived") #[["age"]].mul(-1) #.fillna(0)
y = df["survived"]

# init_preds = np.repeat(0.5, y.shape)
# g = ll.grad(y, init_preds)
# h = ll.hess(y, init_preds)
# t.fit(X.to_numpy(), grad=g, hess=h)
# print(t)
# print(t.predict(X.head().to_numpy()))
# print(len(t.nodes_))
# print(t.predict(X.to_numpy()[0:10,:]))
# print(len(t.nodes_))
# print(t.nodes_[0])
# print(t.nodes_[1].weight_value)
# print(t.nodes_[2].weight_value)
# print(t.predict(X.to_numpy()[0:10]))

# w = np.ones(y.shape)
# # # w[X["pclass"].eq(2)] = 4
# m = XGBoost(
#     iterations=10,
#     learning_rate=0.3,
#     max_depth=5,
#     l2=1,
#     min_leaf_weight=1,
#     gamma=0,
# )
# # m = XGBoost(
# #     iterations=10,
# #     max_depth=5,
# #     min_leaf_weight=1,
# #     gamma=0,
# # )
# m.fit(X.to_numpy(), y, sample_weight=w)
# mp = m.predict(X.to_numpy())
# # # len(m.trees_[0].nodes_)
# # print(m.trees_[0])
# print(len(m.trees_[0].nodes_))
# print(len(m.trees_[-1].nodes_))

# # # print(len(m.trees_[0].__repr__().split()))
# print(mp[0:10])
# print(m.trees_[0])

from xgboost import XGBClassifier
xmod = XGBClassifier(
    n_estimators=10, 
    learning_rate=0.3,
    max_depth=3,
    reg_lambda=1,
    min_child_weight=1,
    gamma=0,
    objective="binary:logitraw",
    eval_metric="auc",
    tree_method="hist",
)
xmod.fit(X, y)
print(xmod.get_booster().get_dump(with_stats=True)[0])
print(xmod.predict(X, output_margin=True)[0:10])

xmod = XGBClassifier(
    n_estimators=1, 
    learning_rate=0.3,
    max_depth=3,
    reg_lambda=1,
    min_child_weight=1,
    gamma=0,
    objective="binary:logitraw",
    eval_metric="auc",
)
xmod.fit(X.fillna(0), y)
print(xmod.get_booster().get_dump(with_stats=True)[0])
print(xmod.predict(X, output_margin=True)[0:10])

# mx = XGBClassifier(seed=123,
#         objective="binary:logitraw",
#         tree_method="exact", #"approx", # "hist",
#         eval_metric="auc",
#         reg_lambda=1,
#         gamma=0,
#         min_child_weight=1,
#         max_leaves=0,
#         max_depth=5,
#         n_estimators=10)
# mx.fit(X, y, sample_weight=w)
# xp = mx.predict(X, output_margin=True)
# print(xp[0:10])

# np.all(np.isclose(mp, xp, rtol=0.00001))


# def log_loss(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
#     y = dtrain.get_label()
#     # predt = 1/(1+np.exp(-predt))
#     # print(predt)
#     # init_preds = np.repeat(0.5, y.shape)
#     g = ll.grad(y, predt)
#     h = ll.hess(y, predt)
#     # print(h.sum())
#     # print(h[0:10])
#     # print(np.unique(y))
#     return g, h


# When testing, know that gamma
# does not seem to behave the same
# for the exact method, as
# it does for the "aprox"
# tree method
# w = np.random.uniform(0.001, 30, size=y.shape)
# d = xgb.DMatrix(X, label=y, weight=y+1)
# mod = xgb.train(
#     dict(
#         seed=123,
#         # objective="binary:logitraw",
#         tree_method="exact", #"approx", # "hist",
#         eval_metric="auc",
#         reg_lambda=1,
#         gamma=0,
#         min_child_weight=1,
#         max_leaves=0,
#         max_depth=5,
#         verbosity=0,
#     ),
#     obj=log_loss,
#     num_boost_round=10,
#     dtrain=d,
#     evals=[(d, "train")],
# )
# mod.get_score(importance_type="gain")
# y[X["pclass"].lt(3)].mean(), y[X["pclass"].ge(3)].mean()
# # print(mod.get_dump(with_stats=True)[0])
# xp = mod.predict(xgb.DMatrix(X, label=y))
# print(xp[0:10])


# print(len(mod.get_dump(with_stats=True)[0].split()))

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
#          "seed=123,
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
