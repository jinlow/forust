# # import pandas as pd
# # import numpy as np
# # from forust import rust_bin_matrix

# # df = pd.read_csv(
# #     "../resources/titanic.csv"
# # )  # .sample(1_000_000, replace=True, random_state=0)

# # X = (
# #     df.select_dtypes("number").drop(columns="survived").reset_index(drop=True)
# # )  # [["pclass"]].astype(float)
# # X_vec = X.to_numpy().ravel(order="F")

# # w = np.ones(X.shape[0])  # np.random.uniform(0.1, 100, X.shape[0]),
# # # w[X["age"] > 56] = 100000

# # b, c, n = rust_bin_matrix(X_vec, X.shape[0], X.shape[1], w, nbins=5)
# # Xb = pd.DataFrame(b.reshape(X.shape, order="F"))
# # c[-1]

# # X_rs = X.copy()
# # for i in range(X.shape[1]):
# #     X_rs.iloc[:, i] = pd.cut(X.iloc[:, i], c[i], right=False).cat.add_categories(
# #         0
# #     )  # pd.Series(np.digitize(X.iloc[:,i], c[i], right=True))
# #     X_rs.iloc[X.iloc[:, i].isna(), i] = 0

# # for i in range(X.shape[1]):
# #     X_rs.iloc[:, i].value_counts()
# #     Xb.iloc[:, i].value_counts()

# # print(X_rs.iloc[:, 0].value_counts().sort_index())
# # print(Xb.iloc[:, 0].value_counts().sort_index())

# # print(X_rs.iloc[:, -1].value_counts().sort_index())
# # print(Xb.iloc[:, -1].value_counts().sort_index())

# # (X_rs.rename(columns={k: i for i, k in enumerate(X_rs.columns)}) == Xb).all()

# # pd.concat([Xb.iloc[:, -1].head(), X.iloc[:, -1].head()], axis=1)
# # c[-1]
# # ####
# # import pandas as pd
# # import numpy as np
# # from forust import percentiles

# # df = pd.read_csv("../resources/titanic.csv").sample(
# #     2_000_000, replace=True, random_state=0
# # )
# # pcts = np.array([0.1, 0.2, 0.5, 0.78])
# # pcts = np.linspace(0, 1, num=200, endpoint=True)
# # p1 = percentiles(df["fare"].to_numpy(), np.ones(df.shape[0]), pcts)


# # df["fare"].nunique()
# # p2 = np.percentile(df["fare"], pcts * 100)
# # np.allclose(p1, p2)

# # ####
# # import pandas as pd
# # import numpy as np
# # from forust import GradientBooster

# # df = pd.read_csv(
# #     "../resources/titanic.csv"
# # )  # .sample(100_000, replace=True, random_state=0)
# # # for i in range(0, 50):
# # i = 1000
# # X = df.select_dtypes("number").drop(columns="survived").reset_index(drop=True)
# # X["age"] = X["age"]  # .mul(-1)
# # y = df["survived"].to_numpy().astype("float64")
# # mod = GradientBooster(
# #     iterations=i,
# #     learning_rate=0.3,
# #     max_depth=5,
# #     l2=1,
# #     min_leaf_weight=1,
# #     gamma=1,
# #     objective_type="LogLoss",
# #     nbins=500,
# #     parallel=False,
# # )
# # mod.fit(X, y)
# # # print(mod.predict(X_vec, y.shape[0], 5)[0:10])

# # from sklearn.ensemble import HistGradientBoostingClassifier

# # # hgb = HistGradientBoostingClassifier(max_iter=100)
# # # hgb.fit(X, y)
# # # hgb.predict_proba(X)[0:10]

# # from xgboost import XGBClassifier, XGBRegressor

# # xmod = XGBClassifier(
# #     n_estimators=i,
# #     learning_rate=0.3,
# #     max_depth=5,
# #     reg_lambda=1,
# #     min_child_weight=1,
# #     gamma=1,
# #     objective="binary:logitraw",
# #     eval_metric="auc",
# #     tree_method="hist",
# #     max_bin=10000,
# # )
# # xmod.fit(X, y)
# # # print(xmod.predict(X, output_margin=True)[0:10])
# # np.allclose(
# #     xmod.predict(X, output_margin=True).astype(np.float64),
# #     mod.predict(X).astype(np.float32),
# #     rtol=0.0001,
# # )
# # # if not np.allclose(xmod.predict(X, output_margin=True).astype(np.float32), mod.predict(X_vec, y.shape[0], 5).astype(np.float32), rtol=0.1):
# # #     print(i)
# # #     break
# # mp = mod.predict(X)
# # xp = xmod.predict(X, output_margin=True)
# # print(mp[0:5])
# # print(xp[0:5])

# # mp[~np.isclose(mp, xp, rtol=0.0001)]
# # xp[~np.isclose(mp, xp, rtol=0.0001)]


# # print(xmod.get_booster().get_dump(with_stats=True)[-1])
# # print(mod.text_dump()[-1])

# import numpy as np

# ### Testing weights
# import pandas as pd
# from xgboost import XGBClassifier

# from forust import GradientBooster

# df = (
#     pd.read_csv("../resources/titanic.csv")
#     .sample(500_000, replace=True, random_state=0)
#     .reset_index(drop=True)
# )
# X = df.select_dtypes("number").drop(columns="survived").reset_index(drop=True)
# y = df["survived"]

# # X = X.fillna(0)
# # w = X["fare"].to_numpy() + 1
# xmod = XGBClassifier(
#     n_estimators=100,
#     learning_rate=0.3,
#     max_depth=5,
#     reg_lambda=1,
#     min_child_weight=1.0,
#     gamma=0.0,
#     objective="binary:logitraw",
#     eval_metric="auc",
#     tree_method="hist",
#     # max_bin=1000,
# )
# xmod.fit(X, y)  # , sample_weight=w)
# xmod_preds = xmod.predict(X, output_margin=True)

# fmod = GradientBooster(
#     iterations=100,
#     learning_rate=0.3,
#     max_depth=5,
#     l2=1,
#     min_leaf_weight=1.0,
#     gamma=0.0,
#     objective_type="LogLoss",
# )
# fmod.fit(X, y=y)  # , sample_weight=w)
# fmod_preds = fmod.predict(X)

# print(fmod_preds[0:10])
# print(xmod_preds[0:10])
# print(np.allclose(fmod_preds, xmod_preds, atol=0.0001))


# fmod.save_booster("mod2.json")
# fmod2 = GradientBooster.load_booster("mod2.json")
# fmod2.predict(X)


# # from sklearn.metrics import roc_auc_score
# # print(roc_auc_score(y, fmod_preds))
# # print(roc_auc_score(y, xmod_preds))

# # from forust import GradientBoosterF32
# # import sys
# # flat_data = X.to_numpy().ravel(order="F")
# # mod = GradientBoosterF32(
# #     objective_type = "LogLoss",
# #         iterations = 100,
# #         learning_rate = 0.3,
# #         max_depth = 5,
# #         max_leaves = sys.maxsize,
# #         l2 = 1.0,
# #         gamma = 0.0,
# #         min_leaf_weight = 0.0,
# #         base_score = 0.5,
# #         nbins = 256,
# #         parallel = True
# # )
# # mod.fit(
# #     flat_data, *X.shape, y.to_numpy(), np.ones(y.shape, "float32")
# # )

# # print(fmod.text_dump()[0])
# # print(xmod.get_booster().get_dump(with_stats=True)[0])


# fmod_preds[~np.isclose(fmod_preds, xmod_preds, rtol=0.001)]
# xmod_preds[~np.isclose(fmod_preds, xmod_preds, atol=0.001)]
# fmod_preds[~np.isclose(fmod_preds, xmod_preds, atol=0.001)]
# assert np.allclose(fmod_preds, xmod_preds, rtol=0.001)


# import numpy as np

# # Trying regression
# import pandas as pd
# from xgboost import XGBClassifier

# from forust import GradientBooster

# df = (
#     pd.read_csv("../resources/titanic.csv")
#     .sample(100_000, replace=True, random_state=0)
#     .reset_index(drop=True)
# )
# X = df.select_dtypes("number").drop(columns="fare").reset_index(drop=True)
# y = df["fare"]

# # X = X.fillna(0)
# # w = X["fare"].to_numpy() + 1
# from xgboost import XGBRegressor

# xmod = XGBRegressor(
#     n_estimators=100,
#     learning_rate=0.3,
#     max_depth=5,
#     reg_lambda=1,
#     min_child_weight=1.0,
#     gamma=0.0,
#     objective="reg:squarederror",
#     eval_metric="auc",
#     # tree_method="hist",
#     # max_bin=1000,
# )
# xmod.fit(X, y)  # , sample_weight=w)
# xmod_preds = xmod.predict(X, output_margin=True)

# fmod = GradientBooster(
#     iterations=100,
#     learning_rate=0.3,
#     max_depth=5,
#     l2=1,
#     min_leaf_weight=1.0,
#     gamma=0.0,
#     objective_type="SquaredLoss",
# )
# fmod.fit(X, y=y)  # , sample_weight=w)
# fmod_preds = fmod.predict(X)

# print(fmod_preds[0:10])
# print(xmod_preds[0:10])

# fmod.save_booster("mod2.json")
# fmod2 = GradientBooster.load_booster("mod2.json")
# fmod2.predict(X)

# fmod.booster.json_dump()
