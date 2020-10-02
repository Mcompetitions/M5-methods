M5 Uncertainty Winning Submission

All models were run in the Kaggle notebook environment with 4 vCPU / 16 GB RAM.

The only package required is !pip install lightgbm. 

--

The notebook contains instructions for replicating the solution, and has been revised to include a version with near-idential performance but ~20x faster run-time for both training and inference.
 
--

/original - original Kaggle code for one training run (of 45), and one inference run (of 12), used for submission
            also includes trained models

/replication - code for replications run on both AWS Sagemaker and Kaggle, using SPEED = True over 5 hours; 1% higher error;
               includes trained models and submission csvs

quantiles_aws.ipynb - training and inference run done on AWS Sagemaker - Datascience Kernel
quantiles_kaggle.ipynb - training and inference Kaggle notebook

join_submission_csvs.ipynb - joins csv files;



