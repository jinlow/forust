<p align="center">
  <img  height="340" src="https://github.com/jinlow/forust/raw/main/resources/tree-image-crop.png">
</p>


<div align="center">

  <a href="https://pypi.org/project/forust/">![PyPI](https://img.shields.io/pypi/v/forust?color=gr&style=for-the-badge)</a>
  <a href="https://crates.io/crates/forust-ml">![Crates.io](https://img.shields.io/crates/v/forust-ml?color=gr&style=for-the-badge)</a>

</div>

# Forust
## _A lightweight gradient boosting package_
Forust, is a lightweight package for building gradient boosted decision tree ensembles. All of the algorithm code is written in [Rust](https://www.rust-lang.org/), with a python wrapper. The rust package can be used directly, however, most examples shown here will be for the python wrapper. For a self contained rust example, [see here](rs-example.md). It implements the same algorithm as the [XGBoost](https://xgboost.readthedocs.io/en/stable/) package, and in many cases will give nearly identical results.

I developed this package for a few reasons, mainly to better understand the XGBoost algorithm, additionally to have a fun project to work on in rust, and because I wanted to be able to experiment with adding new features to the algorithm in a smaller simpler codebase.

All of the rust code for the package can be found in the [src](src/) directory, while all of the python wrapper code is in the [py-forust](py-forust/) directory.

## Installation
The package can be installed directly from [pypi](https://pypi.org/project/forust/).
```shell
pip install forust
```

To use in a rust project add the following to your Cargo.toml file.
```toml
forust-ml = "0.2.0"
```

## Usage
The `GradientBooster` class is currently the only public facing class in the package, and can be used to train gradient boosted decision tree ensembles with multiple objective functions.

It can be initialized with the following arguments.

 - `objective_type` ***(str, optional)***: The name of objective function used to optimize.
    Valid options include "LogLoss" to use logistic loss as the objective function (binary classification),
    or "SquaredLoss" to use Squared Error as the objective function (continuous regression).
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
    required to be in a node. Defaults to 1.0.
 - `base_score` ***(float, optional)***: The initial prediction value of the model. Defaults to 0.5.
 - `nbins` ***(int, optional)***: Number of bins to calculate to partition the data. Setting this to
    a smaller number, will result in faster training time, while potentially sacrificing
    accuracy. If there are more bins, than unique values in a column, all unique values
    will be used. Defaults to 256.
 - `parallel` ***(bool, optional)***: Should multiple cores be used when training and predicting
    with this model? Defaults to `True`.
 - `allow_missing_splits` ***(bool, optional)***: Allow for splits to be made such that all missing values go down one branch, and all non-missing values go down the other, if this results in the greatest reduction of loss. If this is false, splits will only be made on non missing values. Defaults to `True`.
 - `monotone_constraints` ***(dict[Any, int], optional)***: Constraints that are used to enforce a specific relationship between the training features and the target variable. A dictionary should be provided where the keys are the feature index value if the model will be fit on a numpy array, or a feature name if it will be fit on a pandas Dataframe. The values of the dictionary should be an integer value of -1, 1, or 0 to specify the relationship that should be estimated between the respective feature and the target variable. Use a value of -1 to enforce a negative relationship, 1 a positive relationship, and 0 will enforce no specific relationship at all. Features not included in the mapping will not have any constraint applied. If `None` is passed no constraints will be enforced on any variable.  Defaults to `None`.

### Training and Predicting

Once, the booster has been initialized, it can be fit on a provided dataset, and performance field. After fitting, the model can be used to predict on a dataset.
In the case of this example, the predictions are the log odds of a given record being 1.

```python
# Small example dataset
from seaborn import load_dataset

df = load_dataset("titanic")
X = df.select_dtypes("number").drop(columns=["survived"])
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
 - `y` ***(ArrayLike)***: Either a pandas Series, or a 1 dimensional numpy array. If "LogLoss" was
   the objective type specified, then this should only contain 1 or 0 values, where 1 is the positive class being predicted. If "SquaredLoss" is the objective type, then any continuous variable can be
   provided.
 - `sample_weight` ***(Optional[ArrayLike], optional)***: Instance weights to use when
    training the model. If None is passed, a weight of 1 will be used for every record.
    Defaults to None.

The predict method accepts the following arguments.
 - `X` ***(FrameLike)***: Either a pandas DataFrame, or a 2 dimensional numpy array, with numeric data.
 - `parallel` ***(Optional[bool], optional)***: Optionally specify if the predict
   function should run in parallel on multiple threads. If `None` is
   passed, the `parallel` attribute of the booster will be used.
   Defaults to `None`.

### Inspecting the Model

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

The `json_dump` method performs the same action, but returns the model as a json representation rather than a text string.

To see an estimate for how a given feature is used in the model, the `partial_dependence` method is provided. This method calculates the partial dependence values of a feature. For each unique value of the feature, this gives the estimate of the predicted value for that feature, with the effects of all features averaged out. This information gives an estimate of how a given feature impacts the model.

The `partial_dependence` method takes the following parameters...

 - `X` ***(FrameLike)***: Either a pandas DataFrame, or a 2 dimensional numpy array.
      This should be the same data passed into the models fit, or predict,
      with the columns in the same order.
 - `feature` ***(Union[str, int])***: The feature for which to calculate the partial
      dependence values. This can be the name of a column, if the provided
      X is a pandas DataFrame, or the index of the feature.

This method returns a 2 dimensional numpy array, where the first column is the sorted unique values of the feature, and then the second column is the partial dependence values for each feature value.

This information can be plotted to visualize how a feature is used in the model, like so.

```python
from seaborn import lineplot
import matplotlib.pyplot as plt

pd_values = model.partial_dependence(X=X, feature="age")

fig = lineplot(x=pd_values[:,0], y=pd_values[:,1],)
plt.title("Partial Dependence Plot")
plt.xlabel("Age")
plt.ylabel("Log Odds")
```
<img  height="340" src="https://github.com/jinlow/forust/raw/main/resources/pdp_plot_age.png">

We can see how this is impacted if a model is created, where a specific constraint is applied to the feature using the `monotone_constraint` parameter.

```python
model = GradientBooster(
    objective_type="LogLoss",
    monotone_constraints={"age": -1},
)
model.fit(X, y)

pd_values = model.partial_dependence(X=X, feature="age")
fig = lineplot(
    x=pd_values[:, 0],
    y=pd_values[:, 1],
)
plt.title("Partial Dependence Plot with Monotonicity")
plt.xlabel("Age")
plt.ylabel("Log Odds")
```
<img  height="340" src="https://github.com/jinlow/forust/raw/main/resources/pdp_plot_age_mono.png">

### Saving the model
To save and subsequently load a trained booster, the `save_booster` and `load_booster` methods can be used. Each accepts a path, which is used to write the model to. The model is saved and loaded as a json object.

```python
trained_model.save_booster("model_path.json")

# To load a model from a json path.
loaded_model = GradientBooster.load_model("model_path.json")
```
