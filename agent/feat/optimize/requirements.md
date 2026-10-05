Review this package and identify speed bottlenecks, and ways that we can optimize, this should not come from purely from code review and speculation, this should come from running benchmarks on the actual package library.

When this package is used in the wild, evaluation datasets are commonly used as well, this should be considered in the optimizations.

If you are able to use flamegraph for understanding where the bottlenecks are that's even better.

Additionally we should identify if the benchmarks should be updated to better identify performance bottlenecks. The python tests are the true validation safetey net. Those should continue to pass with the parity check against XGBoost.

Please run some initial testing and review the code to determine high performance changes that could be made, create a recomendation markdown document, that we can look at and weigh any pros and cons for speed improvments.