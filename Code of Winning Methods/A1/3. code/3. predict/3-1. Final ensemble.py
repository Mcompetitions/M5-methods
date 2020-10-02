#!/usr/bin/env python
# coding: utf-8

# ## Please input your directory for the top level folder
# folder name : SUBMISSION MODEL

# In[ ]:


dir_ = '/home/artemis/M5/A1-Yeon/' # input only here


# #### setting other directory

# In[ ]:


raw_data_dir = dir_+'2. data/'
processed_data_dir = dir_+'2. data/processed/'
log_dir = dir_+'4. logs/'
model_dir = dir_+'5. models/'
submission_dir = dir_+'6. submissions/'


# In[ ]:


import pandas as pd


# In[ ]:


submission = pd.read_csv(raw_data_dir+'sample_submission.csv')
ids = pd.DataFrame({'id':submission.iloc[30490:]['id']})


# In[ ]:


sub1= pd.read_csv(submission_dir+'before_ensemble/submission_kaggle_recursive_store.csv')
sub2= pd.read_csv(submission_dir+'before_ensemble/submission_kaggle_recursive_store_cat.csv')
sub3= pd.read_csv(submission_dir+'before_ensemble/submission_kaggle_recursive_store_dept.csv')

sub4= pd.read_csv(submission_dir+'before_ensemble/submission_kaggle_nonrecursive_store.csv')
sub5= pd.read_csv(submission_dir+'before_ensemble/submission_kaggle_nonrecursive_store_cat.csv')
sub6= pd.read_csv(submission_dir+'before_ensemble/submission_kaggle_nonrecursive_store_dept.csv')


# In[ ]:


sub1 = ids.merge(sub1, on='id', how='left').set_index('id')
sub2 = ids.merge(sub2, on='id', how='left').set_index('id')
sub3 = ids.merge(sub3, on='id', how='left').set_index('id')

sub4 = ids.merge(sub4, on='id', how='left').set_index('id')
sub5 = ids.merge(sub5, on='id', how='left').set_index('id')
sub6 = ids.merge(sub6, on='id', how='left').set_index('id')


# In[ ]:


final_sub = (sub1 + sub2 + sub3 + sub4 + sub5 + sub6 )/6


# In[ ]:


final_sub.to_csv(submission_dir+'submission_final.csv')


# In[ ]:


final_sub

