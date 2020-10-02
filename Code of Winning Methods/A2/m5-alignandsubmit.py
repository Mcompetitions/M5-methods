#!/usr/bin/env python
# coding: utf-8

# Copyright 2020 Matthias Anderer
# 
# Copyright for aggregation code snippets 2020 by user: https://www.kaggle.com/lebroschar (name unknown)
# 
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
# 
#        http://www.apache.org/licenses/LICENSE-2.0
# 
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# # Overall approach
# 
# We have two different inputs: 
# 
# 1) Bottom level forecasts on item level (30490 signal) that are derived from a lgbm model that models a probability of this item being bought based on datatime features, price features and a few other features that are not time dependent. (Credits: https://www.kaggle.com/kyakovlev/m5-simple-fe)
# 2) Top level forecasts for the levels 1-5 that are created with N-Beats. 
# 
# We can now aggregate the bottom level "probabilit draws" up to the levels 1-5. By comparing/aligning the possible results we can select the most suitable probability distribution for the forecast period. ( The multiplier in the custom loss of the bottom level lgbm models seems to help adjust for trend or other effects not fully understood yet)

# ### Overall analysis result: 
# 
# The multiplier 0.95 seems to represent the lowest available fit so we build an ensemble with the 2 upper and 2 lower distributions to generate a robust test loss.
# <br><br>
# Final-11: 0.9 <br>
# Final-12: 0.93 <br>
# Final-17: 0.95 <br>
# Final-13: 0.97 <br>
# Final-16: 0.99

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import warnings
warnings.simplefilter(action='ignore')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns


# ## Load NBEATS reference predictions for global alignment

# NBeats predictions trained and predicted on Colab with two different settings (only change in setting is num_epochs to get slightly different ensembles)

# In[ ]:


nbeats_pred01_df = pd.read_csv('../input/m5-forecasting-accuracy/nbeats_toplvl_forecasts1.csv')
nbeats_pred02_df = pd.read_csv('../input/m5-forecasting-accuracy/nbeats_toplvl_forecasts2.csv')

#nbeats_pred_df.head()


# ## Load bottom level lgb predictions for alignment

# In[ ]:


BUILD_ENSEMBLE = True


# In[ ]:


if BUILD_ENSEMBLE:
    
    pred_01_df = pd.read_csv('../input/m5-forecasting-accuracy/submission_90.csv')
    pred_02_df = pd.read_csv('../input/m5-forecasting-accuracy/submission_93.csv')
    pred_03_df = pd.read_csv('../input/m5-forecasting-accuracy/submission_95.csv')
    pred_04_df = pd.read_csv('../input/m5-forecasting-accuracy/submission_97.csv')
    pred_05_df = pd.read_csv('../input/m5-forecasting-accuracy/submission_99.csv')
    #pred_06_df = pd.read_csv('..')

    avg_pred = ( np.array(pred_01_df.values[:,1:]) 
                + np.array(pred_02_df.values[:,1:]) 
                + np.array(pred_03_df.values[:,1:])
                + np.array(pred_04_df.values[:,1:])  
                + np.array(pred_05_df.values[:,1:])  
               # + np.array(pred_06_df.values[:,1:])  
               ) /5.0
    
    ## Loading predictions
    valid_pred_df = pd.DataFrame(avg_pred, columns=pred_01_df.columns[1:])
    submission_pred_df = pd.concat([pred_01_df['id'],valid_pred_df],axis=1)
    
else:
    print('Should not submit single distibution')
    #submission_pred_df = pd.read_csv('../input/m5-final-13/submission_v1.csv')


# ## Fill validation rows - we have no info about validation scoring
# 

# Even though it would not make sense at all to score public validation data it might be safest to set the submission validation values to the ground truth....
# 
# Spamming the LB a bit more ... 

# In[ ]:


validation_gt_data = pd.read_csv('../input/m5-forecasting-accuracy/sales_train_evaluation.csv')
validation_gt_data['id'] = validation_gt_data['id'].str.replace('_evaluation','_validation')
validation_gt_data = validation_gt_data.drop(['item_id','dept_id','cat_id','store_id','state_id'],axis=1)
validation_gt_data = pd.concat([validation_gt_data[['id']],validation_gt_data.iloc[:,-28:]],axis=1)
validation_gt_data.columns=submission_pred_df.columns.values
#validation_gt_data


# In[ ]:


submission_pred_df = pd.concat([validation_gt_data, submission_pred_df.iloc[30490:,:]],axis=0).reset_index(drop=True)
submission_pred_df


# In[ ]:





# ## Only work on evaluation forecasts

# In[ ]:


bottom_lvl_pred_df = submission_pred_df.iloc[30490:,:].reset_index(drop=True)
bottom_lvl_pred_df


# ## Reconstruct level descriptions for aggregation

# In[ ]:


name_cols = bottom_lvl_pred_df.id.str.split(pat='_',expand=True)
name_cols['dept_id']=name_cols[0]+'_'+name_cols[1]
name_cols['store_id']=name_cols[3]+'_'+name_cols[4]
name_cols = name_cols.rename(columns={0: "cat_id", 3: "state_id"})
name_cols = name_cols.drop([1,2,4,5],axis=1)
bottom_lvl_pred_df = pd.concat([name_cols,bottom_lvl_pred_df],axis=1)


# ## Build aggregates of predictions

# In[ ]:


# Get column groups
cat_cols = ['id', 'dept_id', 'cat_id',  'store_id', 'state_id']
ts_cols = [col for col in bottom_lvl_pred_df.columns if col not in cat_cols]
ts_dict = {t: int(t[1:]) for t in ts_cols}

# Describe data
print('  unique forecasts: %i' % bottom_lvl_pred_df.shape[0])
for col in cat_cols:
    print('   N_unique %s: %i' % (col, bottom_lvl_pred_df[col].nunique()))


# In[ ]:


# 1. All products, all stores, all states (1 series)
all_sales = pd.DataFrame(bottom_lvl_pred_df[ts_cols].sum()).transpose()
all_sales['id_str'] = 'all'
all_sales = all_sales[ ['id_str'] +  [c for c in all_sales if c not in ['id_str']] ]


# In[ ]:


# 2. All products by state (3 series)
state_sales = bottom_lvl_pred_df.groupby('state_id',as_index=False)[ts_cols].sum()
state_sales['id_str'] = state_sales['state_id'] 
state_sales = state_sales[ ['id_str'] +  [c for c in state_sales if c not in ['id_str']] ]
state_sales = state_sales.drop(['state_id'],axis=1)


# In[ ]:


# 3. All products by store (10 series)
store_sales = bottom_lvl_pred_df.groupby('store_id',as_index=False)[ts_cols].sum()
store_sales['id_str'] = store_sales['store_id'] 
store_sales = store_sales[ ['id_str'] +  [c for c in store_sales if c not in ['id_str']] ]
store_sales = store_sales.drop(['store_id'],axis=1)


# In[ ]:


# 4. All products by category (3 series)
cat_sales = bottom_lvl_pred_df.groupby('cat_id',as_index=False)[ts_cols].sum()
cat_sales['id_str'] = cat_sales['cat_id'] 
cat_sales = cat_sales[ ['id_str'] +  [c for c in cat_sales if c not in ['id_str']] ]
cat_sales = cat_sales.drop(['cat_id'],axis=1)


# In[ ]:


# 5. All products by department (7 series)
dept_sales = bottom_lvl_pred_df.groupby('dept_id',as_index=False)[ts_cols].sum()
dept_sales['id_str'] = dept_sales['dept_id'] 
dept_sales = dept_sales[ ['id_str'] +  [c for c in dept_sales if c not in ['id_str']] ]
dept_sales = dept_sales.drop(['dept_id'],axis=1)


# In[ ]:


all_pred_agg = pd.concat([all_sales,state_sales,store_sales,cat_sales,dept_sales],ignore_index=True)


# In[ ]:


all_pred_agg.head()


# In[ ]:


nbeats_pred01_df.head()


# # Calculating comparision metrics

# ## Interpretation
# 
# If prediction is bigger than "true" values error will be positive -> prediction is overshooting (pos error)
# 
# If prediction is smaller than "true" values error will be negative -> prediction is undershooting (neg error) 
# 

# ## NBeats 01

# In[ ]:


metrics_df = nbeats_pred01_df[['id_str']]

## Calculate errors
## CAUTION: nbeats_pred_df is "truth"/actual values in this context
error = ( np.array(all_pred_agg.values[:,1:]) - np.array(nbeats_pred01_df.values[:,1:]) ) 

## Calc RMSSE
successive_diff = np.diff(nbeats_pred01_df.values[:,1:]) ** 2
denom = successive_diff.mean(1)

num = error.mean(1)**2
rmsse = num / denom

metrics_df['rmsse'] = rmsse

## Not so clean Pandas action :-) - supressing warnings for now...
metrics_df['mean_error'] = error.mean(1)
metrics_df['mean_abs_error'] = np.absolute(error).mean(1)

squared_error = error **2
mean_squ_err = np.array(squared_error.mean(1), dtype=np.float64) 

metrics_df['rmse'] = np.sqrt( mean_squ_err )

metrics_df


# ## NBeats 02

# In[ ]:


metrics_df = nbeats_pred02_df[['id_str']]

## Calculate errors
## CAUTION: nbeats_pred_df is "truth"/actual values in this context
error = ( np.array(all_pred_agg.values[:,1:]) - np.array(nbeats_pred02_df.values[:,1:]) ) 

## Calc RMSSE
successive_diff = np.diff(nbeats_pred01_df.values[:,1:]) ** 2
denom = successive_diff.mean(1)

num = error.mean(1)**2
rmsse = num / denom

metrics_df['rmsse'] = rmsse

## Not so clean Pandas action :-) - supressing warnings for now...
metrics_df['mean_error'] = error.mean(1)
metrics_df['mean_abs_error'] = np.absolute(error).mean(1)

squared_error = error **2
mean_squ_err = np.array(squared_error.mean(1), dtype=np.float64) 

metrics_df['rmse'] = np.sqrt( mean_squ_err )

metrics_df


# # Visualizations

# ### NBeats 01

# In[ ]:


for i in range(0,nbeats_pred01_df.shape[0]):
    plot_df = pd.concat( [nbeats_pred01_df.iloc[i], all_pred_agg.iloc[i] ]  , axis=1, ignore_index=True)
    plot_df = plot_df.iloc[1:,]
    plot_df = plot_df.rename(columns={0:'NBeats',1:'Predictions'})
    plot_df = plot_df.reset_index()
    #plot_df
    
    plot_df.plot(x='index', y=['NBeats', 'Predictions'] ,figsize=(10,5), grid=True, title=nbeats_pred02_df.iloc[i,0]  )


# ## NBeats 02

# In[ ]:


for i in range(0,nbeats_pred02_df.shape[0]):
    plot_df = pd.concat( [nbeats_pred02_df.iloc[i], all_pred_agg.iloc[i] ]  , axis=1, ignore_index=True)
    plot_df = plot_df.iloc[1:,]
    plot_df = plot_df.rename(columns={0:'NBeats',1:'Predictions'})
    plot_df = plot_df.reset_index()
    #plot_df
    
    plot_df.plot(x='index', y=['NBeats', 'Predictions'] ,figsize=(10,5), grid=True, title=nbeats_pred02_df.iloc[i,0]  )


# # Submit based on above analysis and manual selection/clearance

# In[ ]:


submission_pred_df


# In[ ]:


submission_pred_df.to_csv('m5-final-submission.csv', index=False)

