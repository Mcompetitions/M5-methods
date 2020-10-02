<br />  
  
#### **How to train your model**
cd final  

<br />

##### **Prepare data for lightGBM model**  
* python prepare_data.py  
Read training data from RAW_DATA_DIR (specified in SETTINGS.json)  
Run any preprocessing steps  
Save the processed data to PROCESSED_DATA_DIR (specified in SETTINGS.json)  

<br />

##### **Train models**  
* python train.py  
Read training data both from RAW_DATA_DIR and PROCESSED_DATA_DIR (specified in SETTINGS.json)  
Train and Save model weights to MODELS_DIR (specified in SETTINGS.json)  

<br />

##### **Make predictions - submission files **  
* python predict.py  
**Predictions for M5 Accuracy competition, used as starting point to M5 uncertainty competition.**  
Read test data both from RAW_DATA_DIR and PROCESSED_DATA_DIR (specified in SETTINGS.json)  
Load models from MODELS_DIR (specified in SETTINGS.json)  
Use models to make predictions on new samples  
Save predictions with filename 'lgbm3keras1.csv.gz' to SUBMISSION_DIR (specified in SETTINGS.json)  

* python predict_uncertainty.py lgbm3keras1.csv.gz  
**Predictions for M5 uncertainty competition.**  
lgbm3keras1.csv.gz is the output of predict.py, located in SUBMISSION_DIR (specified in SETTINGS.json). Placing another file inside submissions folder and calling predict_uncertainty.py on that file will generate a different submission file (same name).  
Read test data from RAW_DATA_DIR (specified in SETTINGS.json)  
Load models M5 accuracy submission file placed in SUBMISSION_DIR (specified in SETTINGS.json)  
Make predictions on new samples  
Save predictions to SUBMISSION_DIR (specified in SETTINGS.json) 


<br />





