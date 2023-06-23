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
forust-ml = "0.2.15"
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
 - `base_score` ***(float | None, optional)***: The initial prediction value of the model. If set to None the parameter `initialize_base_score` will automatically be set to `True`, in which case the base score will be chosen based on the objective function at fit time. Defaults to `None`.
 - `nbins` ***(int, optional)***: Number of bins to calculate to partition the data. Setting this to
    a smaller number, will result in faster training time, while potentially sacrificing
    accuracy. If there are more bins, than unique values in a column, all unique values
    will be used. Defaults to 256.
 - `parallel` ***(bool, optional)***: Should multiple cores be used when training and predicting
    with this model? Defaults to `True`.
 - `allow_missing_splits` ***(bool, optional)***: Allow for splits to be made such that all missing values go down one branch, and all non-missing values go down the other, if this results in the greatest reduction of loss. If this is false, splits will only be made on non missing values. If `create_missing_branch` is set to `True` having this parameter be set to `True` will result in the missing branch further split, if this parameter is `False` then in that case the missing branch will always be a terminal node. Defaults to `True`.
 - `monotone_constraints` ***(dict[Any, int], optional)***: Constraints that are used to enforce a specific relationship between the training features and the target variable. A dictionary should be provided where the keys are the feature index value if the model will be fit on a numpy array, or a feature name if it will be fit on a pandas Dataframe. The values of the dictionary should be an integer value of -1, 1, or 0 to specify the relationship that should be estimated between the respective feature and the target variable. Use a value of -1 to enforce a negative relationship, 1 a positive relationship, and 0 will enforce no specific relationship at all. Features not included in the mapping will not have any constraint applied. If `None` is passed no constraints will be enforced on any variable.  Defaults to `None`.
 - `subsample` ***(float, optional)***: Percent of records to randomly sample at each iteration when
      training a tree. Defaults to 1.0, meaning all data is used for training.
 - `top_rate` ***(float, optional)***: Used only in goss. The retain ratio of large gradient data. Defaults to 0.1.
 - `other_rate` ***(float, optional)***: Used only in goss. the retain ratio of small gradient data. Defaults to 0.2.
 - `seed` ***(integer, optional)***: Integer value used to seed any randomness used in the
      algorithm. Defaults to 0.
 - `missing` ***(float, optional)***: Value to consider missing, when training and predicting with the booster. Defaults to `np.nan`.
 - `create_missing_branch` ***(bool, optional)***: An experimental parameter, that if `True`, will create a separate branch for missing, creating a ternary tree, the missing node will be given the same weight value as the parent node. If this parameter is `False`, missing will be sent down either the left or right branch, creating a binary tree. Defaults to `False`.
 - `sample_method` ***(str | None, optional)***: Optional string value to use to determine the method to use to sample the data while training. If this is None, no sample method will be used. If the `subsample` parameter is less than 1 and no sample_method is provided this `sample_method` will be automatically set to "random". Valid options are "goss" and "random". Defaults to `None`.
 - `grow_policy` ***(str, optional)***: Optional string value that controls the way new nodes are added to the tree. Choices are `DepthWise` to split at nodes closest to the root, or `LossGuide` to split at nodes with the highest loss change.
 - `evaluation_metric` ***(str | None, optional)***: Optional string value used to define an evaluation metric that will be calculated at each iteration if a `evaluation_dataset` is provided at fit time. The metric can be one of "AUC", "LogLoss", "RootMeanSquaredLogError", or "RootMeanSquaredError". If no `evaluation_metric` is passed, but an `evaluation_dataset` is passed, then "LogLoss", will be used with the "LogLoss" objective function, and "RootMeanSquaredLogError" will be used with "SquaredLoss".
 - `early_stopping_rounds` ***(int | None, optional)***: If this is specified, and an `evaluation_dataset` is passed during fit, then an improvement in the `evaluation_metric` must be seen after at least this many iterations of training, otherwise training will be cut short.
 - `initialize_base_score` (bool, optional): If this is specified, the `base_score` will be calculated using the sample_weight and y data in accordance with the requested `objective_type`.

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

# predict contributions
model.predict_contributions(X.head())
# array([[-0.63014213,  0.33880048, -0.16520798, -0.07798772, -0.85083578,
#        -1.07720813],
#       [ 1.05406709,  0.08825999,  0.21662544, -0.12083538,  0.35209258,
#        -1.07720813],
```

The `fit` method accepts the following arguments.
 - `X` ***(FrameLike)***: Either a pandas DataFrame, or a 2 dimensional numpy array, with numeric data.
 - `y` ***(ArrayLike)***: Either a pandas Series, or a 1 dimensional numpy array. If "LogLoss" was
   the objective type specified, then this should only contain 1 or 0 values, where 1 is the positive class being predicted. If "SquaredLoss" is the objective type, then any continuous variable can be provided.
 - `sample_weight` ***(Optional[ArrayLike], optional)***: Instance weights to use when training the model. If None is passed, a weight of 1 will be used for every record. Defaults to None.
 - `evaluation_data` ***(tuple[FrameLike, ArrayLike, ArrayLike] | tuple[FrameLike, ArrayLike], optional)***: An optional list of tuples, where each tuple should contain a dataset, and equal length target array, and optional an equal length sample weight array. If this is provided metric values will be calculated at each iteration of training. If `early_stopping_rounds` is supplied, the first entry of this list will be used to determine if performance has improved over the last set of iterations, for which if no improvement is not seen in `early_stopping_rounds` training will be cut short.

The predict method accepts the following arguments.
 - `X` ***(FrameLike)***: Either a pandas DataFrame, or a 2 dimensional numpy array, with numeric data.
 - `parallel` ***(Optional[bool], optional)***: Optionally specify if the predict function should run in parallel on multiple threads. If `None` is passed, the `parallel` attribute of the booster will be used. Defaults to `None`.

The `predict_contributions` method will predict with the fitted booster on new data, returning the feature contribution matrix. The last column is the bias term.
 - `X` ***(FrameLike)***: Either a pandas DataFrame, or a 2 dimensional numpy array, with numeric data.
 - `method` ***(str, optional)***: Method to calculate the contributions, the options are.
      - "average": If this option is specified, the average internal node values are calculated, this is equivalent to the `approx_contribs` parameter in XGBoost.
      - "weight": This method will use the internal leaf weights, to calculate the contributions. This is the same as what is described by Saabas [here](https://blog.datadive.net/interpreting-random-forests/).
      - "branch-difference": This method will calculate contributions by subtracting the weight of the node the record will travel down by the weight of the other non-missing branch. This method does not have the property where the contributions summed is equal to the final prediction of the model.
      - "midpoint-difference": This method will calculate contributions by subtracting the weight of the node the record will travel down by the mid-point between the right and left node weighted by the cover of each node. This method does not have the property where the contributions summed is equal to the final prediction of the model.
      - "mode-difference": This method will calculate contributions by subtracting the weight of the node the record will travel down by the weight of the node with the largest cover (the mode node). This method does not have the property where the contributions summed is equal to the final prediction of the model.
 - `parallel` ***(Optional[bool], optional)***: Optionally specify if the predict function should run in parallel on multiple threads. If `None` is passed, the `parallel` attribute of the booster will be used. Defaults to `None`.

When predicting with the data, the maximum iteration that will be used when predicting can be set using the `set_prediction_iteration` method. If `early_stopping_rounds` has been set, this will default to the best iteration, otherwise all of the trees will be used. It accepts a single value.
 - `iteration` (int): Iteration number to use, this will use all trees, up to and including this index.

If early stopping was used, the evaluation history can be retrieved with the `get_evaluation_history` method.

```python
model = GradientBooster(objective_type="LogLoss")
model.fit(X, y, evaluation_data=[(X, y)])

model.get_evaluation_history()[0:3]

# array([[588.9158873 ],
#        [532.01055803],
#        [496.76933646]])
```

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

 - `X` ***(FrameLike)***: Either a pandas DataFrame, or a 2 dimensional numpy array. This should be the same data passed into the models fit, or predict, with the columns in the same order.
 - `feature` ***(Union[str, int])***: The feature for which to calculate the partial dependence values. This can be the name of a column, if the provided X is a pandas DataFrame, or the index of the feature.
 - `samples` ***(int | None, optional)***: Number of evenly spaced samples to select. If None is passed all unique values will be used. Defaults to 100.
 - `exclude_missing` ***(bool, optional)***: Should missing excluded from the features? Defaults to True.
 - `percentile_bounds` ***(tuple[float, float], optional)***: Upper and lower percentiles to start at  when calculating the samples. Defaults to (0.2, 0.98) to cap the samples selected  at the 5th and 95th percentiles respectively.
This method returns a 2 dimensional numpy array, where the first column is the sorted unique values of the feature, and then the second column is the partial dependence values for each feature value.

This information can be plotted to visualize how a feature is used in the model, like so.

```python
from seaborn import lineplot
import matplotlib.pyplot as plt

pd_values = model.partial_dependence(X=X, feature="age", samples=None)

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

Feature importance values can be calculated with the `calculate_feature_importance` method. This function will return a dictionary of the features and their importances. It should be noted that if a feature was never used for splitting it will not be returned in importance dictionary. This function takes the following arguments.
 - `method` ***(str, optional)***: Variable importance method. Defaults to "Gain". Valid options are:
      - "Weight": The number of times a feature is used to split the data across all trees.
      - "Gain": The average split gain across all splits the feature is used in.
      - "Cover": The average coverage across all splits the feature is used in.
      - "TotalGain": The total gain across all splits the feature is used in.
      - "TotalCover": The total coverage across all splits the feature is used in.
 - `normalize` ***(bool, optional)***: Should the importance be normalized to sum to 1? Defaults to `True`.

```python
model.calculate_feature_importance("Gain")
# {
#   'parch': 0.0713072270154953, 
#   'age': 0.11609109491109848,
#   'sibsp': 0.1486879289150238,
#   'fare': 0.14309120178222656,
#   'pclass': 0.5208225250244141
# }
```

### Saving the model
To save and subsequently load a trained booster, the `save_booster` and `load_booster` methods can be used. Each accepts a path, which is used to write the model to. The model is saved and loaded as a json object.

```python
trained_model.save_booster("model_path.json")

# To load a model from a json path.
loaded_model = GradientBooster.load_model("model_path.json")
```
