---
title: "M5 Forecasting - Uncertainty"
author: "Ioannis Nasios"
date: "July 9, 2020"
output: html_document
---

kaggle name: Ouranos  

<br />

Below you can find a outline of how to reproduce my solution.
If you run into any trouble with the setup/code or have any questions please contact me at <nasioannis@hotmail.com>  
 

<br />

#### **HARDWARE: (The following specs were used to create the original solution)**  
CPU: Intel(R) Xeon(R) CPU X5650  @ 2.67GHz with 24 cores  
GPU: GeForce GTX 1080 (server has 2 gpus but I only used 1)  
74 GB memory (used only a fraction)    

<br />
  
#### **OS/platforms:**   
Ubuntu 18.04.2 LTS  

<br />

#### **SOFTWARE (python packages are detailed separately in `requirements.txt`):**
Python 3.6.7  
CUDA 10.0.130  
cuddn 7.5.0  
nvidia drivers version 410.48  

<br />


#### **How to train your model**  
see entry_points.md  


 

  
<br />


#### **How to make predictions on a new test set**
To make predictions on new data, replace raw files with new.   
Raw files:  
- sales_train_evaluation.csv  
- calendar.csv  
- sell_prices.csv  
sample_submission.csv is also needed.   
Second half (rowwise) of submission file correspond to new predictions.  

<br />


#### **Outputs - Submissions**
* **M5 Uncertainty final submission: submission_uncertainty.csv**
* M5 Accuracy submission, used as starting point of M5 Uncertainty: lgbm3keras1.csv.gz  

* M5 Accuracy keras1 submission: Keras_CatEmb_final3_et1941_ver-17last.csv.gz  
* M5 Accuracy keras2 submission: Keras_CatEmb_final3_et1941_ver-EN1EN2Emb1.csv.gz  
* M5 Accuracy keras3 submission: Keras_CatEmb_final3_et1941_ver-noEN1EN2.csv.gz  

* M5 Accuracy average 3 keras submission: Keras_CatEmb_final3_et1941_ver-avg3.csv.gz
* M5 Accuracy lightgbm submission: lgbm_final_VER4.csv.gz  

<br />


#### **Key assumptions made by your code.**  
lgbm_datasets folder must be empty when starting a training run


<br />


