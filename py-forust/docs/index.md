# Forust 
## Python API Reference

<a href="https://pypi.org/project/forust/">![PyPI](https://img.shields.io/pypi/v/forust?color=gr&style=for-the-badge)</a>

<a href="https://crates.io/crates/forust-ml">![Crates.io](https://img.shields.io/crates/v/forust-ml?color=gr&style=for-the-badge)</a>


The `GradientBooster` class is currently the only public facing class in the package, and can be used to train gradient boosted decision tree ensembles with multiple objective functions.


::: forust.GradientBooster

## Logging output

Info is logged while the model is being trained if the `verbose` parameter is set to `True` while fitting the booster. The logs can be printed to stdout while training like so.

```python
import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

fmod = GradientBooster(verbose=True)
fmod.fit(X, y, evaluation_data=[(X, y)])

# INFO:forust_ml.gradientbooster:Iteration 0 evaluation data values: 0.2828
# INFO:forust_ml.gradientbooster:Completed iteration 0 of 10
# INFO:forust_ml.gradientbooster:Iteration 1 evaluation data values: 0.2807
# INFO:forust_ml.gradientbooster:Completed iteration 1 of 10
# INFO:forust_ml.gradientbooster:Iteration 2 evaluation data values: 0.2787
# INFO:forust_ml.gradientbooster:Completed iteration 2 of 10
```

The log output can also be captured in a file also using the `logging.basicConfig()`.

```python
import logging
logging.basicConfig(filename="training-info.log")
logging.getLogger().setLevel(logging.INFO)

fmod = GradientBooster(verbose=True)
fmod.fit(X, y, evaluation_data=[(X, y)])
```
