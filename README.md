# Forust
## _A lightweight gradient boosting package_

Forust, is a lightweight package for building gradient boosted decision tree ensembles. All of the algorithm code is written in [Rust](https://www.rust-lang.org/), with a python wrapper. The rust package can be used directly, however, most examples shown here will be for the python wrapper. It implements the same algorithm as the [XGBoost](https://xgboost.readthedocs.io/en/stable/) package, and in many cases will give nearly identical results.

I developed this package for a few reasons, mainly to better understand the XGBoost algorithm, additionally to have a fun project to work on in rust, and because I wanted to be able to experiment with adding new features to the algorithm in a smaller simpler codebase.

### Usage
The `GradientBooster` class is currently the only public facing class in the package, and can be used to train gradient boosted decision tree ensembles with multiple objective functions.

It can be initialized with the following arguments.

 - `objective_type` ***(str, optional)***: The name of objective function used to optimize.
    Valid options include "LogLoss" to use logistic loss as the objective function,
    or "SquaredLoss" to use Squared Error as the objective function.
    Defaults to "LogLoss".
 - `iterations` ***(int, optional)***: Total number of trees to train in the ensemble.
    Defaults to 100.
 - `learning_rate` ***(float, optional)***: Step size to use at each iteration. Each
    leaf weight is multiplied by this number. The smaller the value, the more
    conservative the weights will be. Defaults to 0.3.
 - `max_depth` ***(int, optional)***: Maximum depth of an individual tree. Valid values
    are 0 to infinity. Defaults to 5.
 - `max_leaves` ***(int, optional)***: Maximum number of leaves allowed on a tree. Valid values
    are 0 to infinity. This is the total number of final nodes. Defaults to sys.maxsize.
 - `l2` ***(float, optional)***: L2 regularization term applied to the weights of the tree. Valid values
    are 0 to infinity. Defaults to 1.0.
 - `gamma` ***(float, optional)***: The minimum amount of loss required to further split a node.
    Valid values are 0 to infinity. Defaults to 0.0.
 - `min_leaf_weight` ***(float, optional)***: Minimum sum of the hessian values of the loss function
    required to be in a node. Defaults to 0.0.
 - `base_score` ***(float, optional)***: The initial prediction value of the model. Defaults to 0.5.
 - `nbins` ***(int, optional)***: Number of bins to calculate to partition the data. Setting this to
    a smaller number, will result in faster training time, while potentially sacrificing
    accuracy. If there are more bins, than unique values in a column, all unique values
    will be used. Defaults to 256.
 - `parallel` ***(bool, optional)***: Should multiple cores be used when training and predicting
    with this model? Defaults to True.
 - `dtype` ***(Union[np.dtype, str], optional)***: Datatype used for the model. Valid options
    are a numpy 32 bit float, or numpy 64 bit float. Using 32 bit float could be faster
    in some instances, however this may lead to less precise results. Defaults to "float64".

Once, the booster has been initialized, it can be fit on a provided dataset, and performance field. After fitting, the model can be used to predict on a dataset.
In the case of this example, the predictions are the log odds of a given record being 1.

```python
# Small example dataset
from seaborn import load_dataset

df = load_dataset("titanic")
X = df.select_dtypes("number").drop(column=["survived"])
y = df["survived"]

# Initialize a booster with defaults.
from forust import GradientBooster
model = GradientBooster(objective_type="LogLoss")
model.fit(X, y)

# Predict on data
model.predict(X.head())
# array([-1.94919663,  2.25863229,  0.32963671,  2.48732194, -3.00371813])
```

The `fit` method accepts the following arguments.
 - `X` ***(FrameLike)***: Either a pandas DataFrame, or a 2 dimensional numpy array, with numeric data.
 - `y` ***(ArrayLike)***: Either a pandas Series, or a 1 dimensional numpy array.
 - `sample_weight` ***(Optional[ArrayLike], optional)***: Instance weights to use when
    training the model. If None is passed, a weight of 1 will be used for every record.
    Defaults to None.

The predict method accepts the following arguments.
 - `X` ***(FrameLike)***: Either a pandas DataFrame, or a 2 dimensional numpy array, with numeric data.

Once the booster has been fit, each individual tree structure can be retrieved in text form, using the `text_dump` method. This method returns a list, the same length as the number of trees in the model.

```python
model.text_dump()[0]
# 0:[0 < 3] yes=1,no=2,missing=2,gain=91.50833,cover=209.388307
#       1:[4 < 13.7917] yes=3,no=4,missing=4,gain=28.185467,cover=94.00148
#             3:[1 < 18] yes=7,no=8,missing=8,gain=1.4576768,cover=22.090348
#                   7:[1 < 17] yes=15,no=16,missing=16,gain=0.691266,cover=0.705011
#                         15:leaf=-0.15120,cover=0.23500
#                         16:leaf=0.154097,cover=0.470007
```

### TODOs
This is still a work in progress
- [ ] Early stopping rounds
    * We should be able to accept a validation dataset, and this should be able to be used to determine when to stop training.
- [ ] Monotonicity support
    * Right now features are used in the model without any constraints.
- [ ] Ability to save a model.
    * The way the underlying trees are structured, they would lend themselves to being saved as JSon objects.
- [ ] Clean up the CICD pipeline.
