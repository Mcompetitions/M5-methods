This project contains 5 jupyter notebooks that must be run in the sequence specified below:

-------------------------------------------------- ------------

1-preprocess-price.ipynb [run in few minutes]:

inputs: "raw_data/sell_prices.csv"

outputs: "proc_data/prices_processed.csv"

This notebook uses the price data provided and performs a processing, returning the processed price data.

-------------------------------------------------- ------------

2-preprocess-calen.ipynb [run in few seconds]:

inputs: "raw_data/calendar.csv"

outputs: "proc_data/processed_calendar.csv"

This notebook uses the calendar data provided and performs a processing, returning the processed calendar data.

-------------------------------------------------- ------------

3-preproc_fast_4.ipynb [run in ~3 hours]

inputs:
"raw_data/sell_prices.csv";
"proc_data/prices_processed.csv";
"raw_data/calendar.csv";
"proc_data/processed_calendar.csv";
"raw_data/sales_train_evaluation.csv";
"raw_data/sample_submission.csv"

outputs: "proc_data/partial_submission.csv"

This notebook performs all the modeling using historical data provided, and processed and unprocessed price and calendar data. In addition, it uses the submission sample to create its own partial submission

Note: by splitting the data between training and validation with random choice, the results can fluctuate.

-------------------------------------------------- ------------

4-compute-mean-error-factors.ipynb [run in few seconds]

inputs:
"raw_data/sales_train_evaluation.csv";
"proc_data/partial_submission.csv"

outputs: "proc_data/factor.csv"

This notebook calculates the difference between the true values ​​and the predicted values ​​of the partial submission on the validation days, and creates a file of factors that will be used to correct the final result
-------------------------------------------------- ------------

5-apply-factors-m5.ipynb [run in few seconds]

inputs:
"proc_data/partial_submission.csv"
"proc_data/factor.csv"
"raw_data/sales_train_evaluation.csv"

outputs: "output_data/submission.csv"

This notebook multiplies the matrix of factors by partial submission, so we have the final result
