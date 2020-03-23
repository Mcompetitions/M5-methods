This R code can be used for implementing the M5 benchmarks for the case of the point forecasts (Accuracy competition) and the probabilistic forecasts (Uncertainty competition), as described in the M5 Competitors' Guide.

The "Point Forecasts - Benchmarks.R" script refers to the point forecasts, while the "Probabilistic Forecasts - Benchmarks.R" to the probabilistic ones.

The code describes - among others - the way the weights are calculated per series, the scaling applied, as well as how the final scores are computed per level and in total.

Given that the validation test-set is not provided to the participants at this stage of the competition, a dummy test-set ("sales_test_validation.csv") is given instead to enable the calculation of the scores.

The rest of the input files are summarized as follows:

1. calendar.csv - Contains information about the dates the products are sold
2. sales_train_validation.rar - Contains the historical daily unit sales data per product and store
3. sell_prices.rar	- Contains information about the price of the products sold per store and date
4. weights_validation.csv - The weights to be used for computing WRMSSE
5. sample_submission.rar	- The Kaggle format for submitting the forecasts
