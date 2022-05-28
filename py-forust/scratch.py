import pandas as pd
import numpy as np
from forust import rust_bin_matrix

df = pd.read_csv("../resources/titanic.csv")# .sample(1_000_000, replace=True, random_state=0)

X = df.select_dtypes("number").drop(columns="survived").reset_index(drop=True) #[["pclass"]].astype(float)
X_vec = X.to_numpy().ravel(order="F")

b, c, n = rust_bin_matrix(X_vec, X.shape[0], X.shape[1], np.ones(X.shape[0]), nbins=5)
Xb = pd.DataFrame(b.reshape(X.shape, order="F"))


X_rs = X.copy()
for i in range(X.shape[1]):
    X_rs.iloc[:,i] = pd.Series(np.digitize(X.iloc[:,i], c[i])).fillna(0)

for i in range(X.shape[1]):
    X_rs.iloc[:,i].value_counts()
    Xb.iloc[:,i].value_counts()

print(X_rs.iloc[:,0].value_counts())
print(Xb.iloc[:,0].value_counts())

print(X_rs.iloc[:,1].value_counts())
print(Xb.iloc[:,1].value_counts())


####
import pandas as pd
import numpy as np
from forust import  percentiles

df = pd.read_csv("../resources/titanic.csv").sample(2_000_000, replace=True, random_state=0)
pcts = np.array([0.1, 0.2, 0.5, 0.78])
pcts = np.linspace(0, 1, num=200, endpoint=True)
p1 = percentiles(df["fare"].to_numpy(), np.ones(df.shape[0]), pcts)


df["fare"].nunique()
p2 = np.percentile(df["fare"], pcts*100)

####
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
