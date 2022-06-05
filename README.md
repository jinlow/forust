# Forust
## _A lightweight gradient boosting package_

Forust, is a lightweight package for building gradient boosted decision tree ensembles. All of the algorithm code is written in [Rust](https://www.rust-lang.org/), with a python wrapper. The rust package can be used directly, however, most examples shown here will be for the python wrapper. It implements the same algorithm as the [XGBoost](https://xgboost.readthedocs.io/en/stable/) package, and in many cases will give nearly identical results.

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

