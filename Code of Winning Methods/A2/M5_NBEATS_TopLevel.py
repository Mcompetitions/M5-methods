#!/usr/bin/env python
# coding: utf-8

# Copyright 2020 Stephan Rabanser, Matthias Anderer
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

# In[ ]:

########### ARTEMIS FIX #################
# Comment out
# from google.colab import drive
# drive.mount('/content/drive')
########### ARTEMIS FIX #################

# In[ ]:


import sys
#package_path = '/content/drive/My Drive/m5data/deepar' 
#sys.path.append(package_path)

package_path = '/gluonts' 
sys.path.append(package_path)

#package_path = '/content/drive/My Drive/m5data/aggregates' 
#sys.path.append(package_path)


###### IF NOT RUN ON COLAB YOU HAVE TO MAKE SURE THAT GLUONTS PACKAGE IS IN YOUR PATH


# # Imports
# 

# In[ ]:


#!pip install pydantic~=1.1 ujson~=1.35
#!pip install --upgrade mxnet-cu101mkl==1.4.1 gluonts --no-deps

########### ARTEMIS FIX #################
# Replace ipython commands
# get_ipython().system('pip install --upgrade pydantic ujson mxnet-cu101mkl==1.4.1 --no-deps')

#!pip install --upgrade pydantic ujson mxnet-cu101mkl --no-deps
import os
exec_command = 'pip uninstall -y gluonts'
os.system(exec_command)
# get_ipython().system('pip uninstall -y gluonts')

########### ARTEMIS FIX #################
# In[ ]:


from gluonts.dataset.common import load_datasets, ListDataset
from gluonts.dataset.field_names import FieldName


# In[ ]:

########### ARTEMIS FIX #################
# Comment out
# get_ipython().run_line_magic('matplotlib', 'inline')
########### ARTEMIS FIX #################
import mxnet as mx
from mxnet import gluon
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from tqdm.autonotebook import tqdm
from pathlib import Path


# In[ ]:


from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.trainer import Trainer
from gluonts.model.n_beats import NBEATSEnsembleEstimator
from gluonts.evaluation import Evaluator


# In[ ]:


class M5Evaluator(Evaluator):
          
        def get_metrics_per_ts(self, time_series, forecast):
              successive_diff = np.diff(time_series.values.reshape(len(time_series)))
              successive_diff = successive_diff ** 2
              successive_diff = successive_diff[:-prediction_length]
              denom = np.mean(successive_diff)
              pred_values = forecast.samples.mean(axis=0)
              true_values = time_series.values.reshape(len(time_series))[-prediction_length:]
              num = np.mean((pred_values - true_values)**2)
              rmsse = num / denom
              metrics = super().get_metrics_per_ts(time_series, forecast)
              metrics["RMSSE"] = rmsse
              return metrics
          
        def get_aggregate_metrics(self, metric_per_ts):
              wrmsse = metric_per_ts["RMSSE"].mean()
              agg_metric , _ = super().get_aggregate_metrics(metric_per_ts)
              agg_metric["MRMSSE"] = wrmsse
              return agg_metric, metric_per_ts


# # Config

# In[ ]:


single_prediction_length = 28
submission_prediction_length = single_prediction_length * 2
m5_input_path="../input/m5-forecasting-accuracy"

SUBMISSION=True
VISUALIZE=False

VERSION=2

CALC_RESIDUALS = False

#if SUBMISSION:
#    prediction_length = submission_prediction_length
#else:
#    prediction_length = single_prediction_length


prediction_length = single_prediction_length


# # Set Seeds

# In[ ]:


# Seed value
seed_value= 247

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)

# 4. Set gluon seed...
mx.random.seed(seed_value)


# # Read Data

# In[ ]:


# Load data
print('Loading data...')
sell_price = pd.read_csv('%s/sell_prices.csv' % m5_input_path)
calendar = pd.read_csv('%s/calendar.csv' % m5_input_path)
train = pd.read_csv('%s/sales_train_evaluation.csv' % m5_input_path).set_index('id')
sample_sub = pd.read_csv('%s/sample_submission.csv' % m5_input_path)


# In[ ]:


#MIN_MEAN_FOR_EXCLUSION=5


# In[ ]:


#train['mean'] = train[train.columns[6:]].mean(axis=1)
#train = train[train['mean'] >= MIN_MEAN_FOR_EXCLUSION]
#train = train.drop(['mean'],axis=1)


# In[ ]:


#train.shape


# # Build aggregate dataset

# In[ ]:


# Get column groups
cat_cols = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
ts_cols = [col for col in train.columns if col not in cat_cols]
ts_dict = {t: int(t[2:]) for t in ts_cols}

# Describe data
print('  unique forecasts: %i' % train.shape[0])
for col in cat_cols:
    print('   N_unique %s: %i' % (col, train[col].nunique()))


# In[ ]:

########### ARTEMIS FIX #################
# get_ipython().run_cell_magic('time', '', "# 1. All products, all stores, all states (1 series)\nall_sales = pd.DataFrame(train[ts_cols].sum()).transpose()\nall_sales['id_str'] = 'all'\nall_sales = all_sales[ ['id_str'] +  [c for c in all_sales if c not in ['id_str']] ]")

# 1. All products, all stores, all states (1 series)
all_sales = pd.DataFrame(train[ts_cols].sum()).transpose()
all_sales['id_str'] = 'all'
all_sales = all_sales[ ['id_str'] +  [c for c in all_sales if c not in ['id_str']] ]


# In[ ]:


# all_sales
########### ARTEMIS FIX #################

# In[ ]:

########### ARTEMIS FIX #################
# get_ipython().run_cell_magic('time', '', "# 2. All products by state (3 series)\nstate_sales = train.groupby('state_id',as_index=False)[ts_cols].sum()\nstate_sales['id_str'] = state_sales['state_id'] \nstate_sales = state_sales[ ['id_str'] +  [c for c in state_sales if c not in ['id_str']] ]\nstate_sales = state_sales.drop(['state_id'],axis=1)")

# 2. All products by state (3 series)
state_sales = train.groupby('state_id',as_index=False)[ts_cols].sum()
state_sales['id_str'] = state_sales['state_id'] 
state_sales = state_sales[ ['id_str'] +  [c for c in state_sales if c not in ['id_str']] ]
state_sales = state_sales.drop(['state_id'],axis=1)

# In[ ]:


# state_sales
########### ARTEMIS FIX #################

# In[ ]:

########### ARTEMIS FIX #################
# get_ipython().run_cell_magic('time', '', "# 3. All products by store (10 series)\nstore_sales = train.groupby('store_id',as_index=False)[ts_cols].sum()\nstore_sales['id_str'] = store_sales['store_id'] \nstore_sales = store_sales[ ['id_str'] +  [c for c in store_sales if c not in ['id_str']] ]\nstore_sales = store_sales.drop(['store_id'],axis=1)")

# 3. All products by store (10 series)
store_sales = train.groupby('store_id',as_index=False)[ts_cols].sum()
store_sales['id_str'] = store_sales['store_id'] 
store_sales = store_sales[ ['id_str'] +  [c for c in store_sales if c not in ['id_str']] ]
store_sales = store_sales.drop(['store_id'],axis=1)

# In[ ]:


# get_ipython().run_cell_magic('time', '', "# 4. All products by category (3 series)\ncat_sales = train.groupby('cat_id',as_index=False)[ts_cols].sum()\ncat_sales['id_str'] = cat_sales['cat_id'] \ncat_sales = cat_sales[ ['id_str'] +  [c for c in cat_sales if c not in ['id_str']] ]\ncat_sales = cat_sales.drop(['cat_id'],axis=1)")

# 4. All products by category (3 series)
cat_sales = train.groupby('cat_id',as_index=False)[ts_cols].sum()
cat_sales['id_str'] = cat_sales['cat_id'] 
cat_sales = cat_sales[ ['id_str'] +  [c for c in cat_sales if c not in ['id_str']] ]
cat_sales = cat_sales.drop(['cat_id'],axis=1)

# In[ ]:


# get_ipython().run_cell_magic('time', '', "# 5. All products by department (7 series)\ndept_sales = train.groupby('dept_id',as_index=False)[ts_cols].sum()\ndept_sales['id_str'] = dept_sales['dept_id'] \ndept_sales = dept_sales[ ['id_str'] +  [c for c in dept_sales if c not in ['id_str']] ]\ndept_sales = dept_sales.drop(['dept_id'],axis=1)")

# 5. All products by department (7 series)
dept_sales = train.groupby('dept_id',as_index=False)[ts_cols].sum()
dept_sales['id_str'] = dept_sales['dept_id'] 
dept_sales = dept_sales[ ['id_str'] +  [c for c in dept_sales if c not in ['id_str']] ]
dept_sales = dept_sales.drop(['dept_id'],axis=1)

# In[ ]:


# get_ipython().run_cell_magic('time', '', "# 6. All products by state and category (9 series)\n# state_cat_sales = pd.read_csv('%s/aggregates/state_cat_sales_agg.csv' % m5_input_path,index_col=False)")
########### ARTEMIS FIX #################

# In[ ]:


# state_cat_sales = state_cat_sales.drop(['Unnamed: 0'],axis=1)


# In[ ]:


# state_cat_sales


# In[ ]:


# get_ipython().run_cell_magic('time', '', "# 7. All products by state and category (21 series) \n# state_dept_sales = pd.read_csv('%s/aggregates/state_dept_sales_agg.csv' % m5_input_path)")


# In[ ]:


# state_dept_sales = state_dept_sales.drop(['Unnamed: 0'],axis=1)


# In[ ]:


# get_ipython().run_cell_magic('time', '', "# 8. All products by store and category (30 series) \n# store_cat_sales = pd.read_csv('%s/aggregates/store_cat_sales_agg.csv' % m5_input_path)\n# store_cat_sales = store_cat_sales.drop(['Unnamed: 0'],axis=1)")


# In[ ]:


# get_ipython().run_cell_magic('time', '', "# 9. All products by store and department (70 series)\n# store_dept_sales = pd.read_csv('%s/aggregates/store_dept_sales_agg.csv' % m5_input_path)\n# store_dept_sales = store_dept_sales.drop(['Unnamed: 0'],axis=1)")


# In[ ]:


# get_ipython().run_cell_magic('time', '', "# 10. all product sales ~3000 signals \n# product_sales = pd.read_csv('%s/aggregates/product_sales_agg.csv' % m5_input_path)\n# product_sales = product_sales.drop(['Unnamed: 0'],axis=1)")


# In[ ]:


#product_sales['sum']=product_sales.iloc[1:].sum(axis=1)
#product_sales


# In[ ]:


# get_ipython().run_cell_magic('time', '', "# 11. all product sales per state ~9000 signals\n# product_state_sales = pd.read_csv('%s/aggregates/product_state_sales_agg.csv' % m5_input_path)\n# product_state_sales = product_state_sales.drop(['Unnamed: 0'],axis=1)")


# In[ ]:





# In[ ]:


#train = train.drop(['item_id','dept_id','store_id','state_id','cat_id'],axis=1)
#train = train.reset_index()
#train = train.rename(columns={'id': 'id_str'})


# In[ ]:


#all_aggregates = pd.concat([all_sales,state_sales,store_sales,cat_sales,dept_sales,state_cat_sales,state_dept_sales,store_cat_sales,store_dept_sales],ignore_index=True)

## TOP LEVEL aggregates + TOTAL
all_aggregates = pd.concat([all_sales,state_sales,store_sales,cat_sales,dept_sales],ignore_index=True)

## MID LEVEL aggregates
#all_aggregates = pd.concat([state_cat_sales,state_dept_sales,store_cat_sales,store_dept_sales],ignore_index=True)

## STATE LEVEL aggregates
#all_aggregates = pd.concat([state_cat_sales,state_dept_sales],ignore_index=True)

#all_aggregates = pd.concat([store_dept_sales],ignore_index=True)

#all_aggregates = product_sales



#### QUICK TESTING FOR NOW bottom level with mean > MIN_MEAN

#all_aggregates = train
#all_aggregates


# # Prepare dataframe for gluon-ts
# 

# In[ ]:


train_df = all_aggregates.drop(["id_str"], axis=1)
train_target_values = train_df.values

if SUBMISSION == True:
    test_target_values = [np.append(ts, np.ones(prediction_length) * np.nan) for ts in train_df.values]
else:
    test_target_values = train_target_values.copy()
    train_target_values = [ts[:-prediction_length] for ts in train_df.values]

m5_dates = [pd.Timestamp("2011-01-29", freq='1D') for _ in range(len(all_aggregates))]

train_ds = ListDataset([
      {
          FieldName.TARGET: target,
          FieldName.START: start
      }
      for (target, start) in zip(train_target_values,
                                          m5_dates
                                          )
  ], freq="D")

test_ds = ListDataset([
      {
          FieldName.TARGET: target,
          FieldName.START: start
      }
      for (target, start) in zip(test_target_values,
                                          m5_dates)
  ], freq="D")


# In[ ]:


num_signals = len(train_df)


# In[ ]:


next(iter(train_ds))


# # Define Estimators and train on aggregates

#     if mode_nbeats:
# 
#     from gluonts.trainer import Trainer
#     from gluonts.model.n_beats import NBEATSEnsembleEstimator
# 
#     estimator = NBEATSEnsembleEstimator(
#         prediction_length=prediction_length,
#         #context_length=7*prediction_length,
#         meta_bagging_size = 1, ## Change back to 10 after testing??
#         meta_context_length = [prediction_length * mlp for mlp in [3,5] ], ## Change back to (2,7)
#         meta_loss_function = ['sMAPE','MASE'], ## Change back to all three MAPE, MASE ...
#         freq="D",
#         trainer=Trainer(
#                       learning_rate=1e-3,
#                       #clip_gradient=1.0,
#                       epochs=15,
#                       num_batches_per_epoch=1000,
#                       batch_size=16
#                       #ctx=mx.context.gpu()
#                   )
# 
#     )

# TOP LEVEL CONFIG
# 
# if True:
# 
#   estimator = NBEATSEnsembleEstimator(
#       prediction_length=prediction_length,
#       #context_length=7*prediction_length,
#       meta_bagging_size = 3,  # 3, ## Change back to 10 after testing??
#       meta_context_length = [prediction_length * mlp for mlp in [3,5,7] ], ## Change back to (2,7) // 3,5,7
#       meta_loss_function = ['sMAPE'], ## Change back to all three MAPE, MASE ...
#       num_stacks = 30,
#       widths= [512],
#       freq="D",
#       trainer=Trainer(
#                     learning_rate=6e-4,
#                     #clip_gradient=1.0,
#                     epochs=10, #10
#                     num_batches_per_epoch=1000,
#                     batch_size=16
#                     #ctx=mx.context.gpu()
#                 )
# 
#   )
#   

# In[ ]:


if True:

  estimator = NBEATSEnsembleEstimator(
      prediction_length=prediction_length,
      #context_length=7*prediction_length,
      meta_bagging_size = 3,  # 3, ## Change back to 10 after testing??
      meta_context_length = [prediction_length * mlp for mlp in [3,5,7] ], ## Change back to (2,7) // 3,5,7
      meta_loss_function = ['sMAPE'], ## Change back to all three MAPE, MASE ...
      num_stacks = 30,
      widths= [512],
      freq="D",
      trainer=Trainer(
                    learning_rate=6e-4,
                    #clip_gradient=1.0,
                    epochs=12, #10
                    num_batches_per_epoch=1000,
                    batch_size=16
                    #ctx=mx.context.gpu()
                )

  )
  


# In[ ]:


if SUBMISSION:
  predictor = estimator.train(train_ds)
else:
  predictor = estimator.train(train_ds,test_ds)


# # Analyze forcasts - Errors and Visual inspection
# 

# In[ ]:


forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_ds,
    predictor=predictor,
    num_samples=100
)

print("Obtaining time series conditioning values ...")
tss = list(tqdm(ts_it, total=len(test_ds)))
print("Obtaining time series predictions ...")
forecasts = list(tqdm(forecast_it, total=len(test_ds)))


# In[ ]:





# In[ ]:


if not SUBMISSION:
      evaluator = M5Evaluator(quantiles=[0.5, 0.67, 0.95, 0.99])
      agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(test_ds))
      print(json.dumps(agg_metrics, indent=4))


# # Visualize forecasts

# In[ ]:


num_series = len(all_aggregates)


# In[ ]:


if VISUALIZE:
  
  plot_log_path = "./plots/"
  directory = os.path.dirname(plot_log_path)
  if not os.path.exists(directory):
      os.makedirs(directory)
      
  def plot_prob_forecasts(ts_entry, forecast_entry, path, sample_id, inline=True):
      plot_length = 150
      prediction_intervals = (50, 99)
      legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]

      _, ax = plt.subplots(1, 1, figsize=(10, 7))
      ts_entry[-plot_length:].plot(ax=ax)
      forecast_entry.plot(prediction_intervals=prediction_intervals, color='g')
      ax.axvline(ts_entry.index[-prediction_length], color='r')
      plt.legend(legend, loc="upper left")
      if inline:
          plt.show()
          plt.clf()
      else:
          plt.savefig('{}forecast_{}.pdf'.format(path, sample_id))
          plt.close()

  print("Plotting time series predictions ...")
  for i in tqdm(range(num_series)):
      ts_entry = tss[i]
      forecast_entry = forecasts[i]
      plot_prob_forecasts(ts_entry, forecast_entry, plot_log_path, i)


# # Get forecast and residuals and write to submission file
# 

# ### In-Sample Residuals
# 
# 1) Loop over prediction_length windows:
# *    create in_sample_test_df
# *   run forecast_it, ts_it = make_evaluation_predictions(
#       dataset=test_ds, ....
# *   Generate residuals data per signal 
# *   write to csv
# 
#     - 

# In[ ]:



if CALC_RESIDUALS:
  all_residuals = []

  # Loop through 10 full prediction length windows to calculate residuals
  for lookback_block in range(1,10):
  #for lookback_block in range(1,2):

      local_train_df = all_aggregates.drop(["id_str"], axis=1)
      local_test_target_values = [ts[:-lookback_block*single_prediction_length] for ts in local_train_df.values]

      ## Startdate is same for all data in this exercise...
      m5_dates = [pd.Timestamp("2011-01-29", freq='1D') for _ in range(len(all_aggregates))]

      local_test_ds = ListDataset([
            {
                FieldName.TARGET: target,
                FieldName.START: start
            }
            for (target, start) in zip(local_test_target_values,
                                                m5_dates)
        ], freq="D")

      local_forecast_it, local_ts_it = make_evaluation_predictions(
            dataset=local_test_ds,
            predictor=predictor,
            num_samples=1
        )
      
      print("Obtaining local time series conditioning values ...")
      local_tss = list(tqdm(local_ts_it, total=len(local_test_ds)))
      print("Obtaining local time series predictions ...")
      in_sample_forecasts = list(tqdm(local_forecast_it, total=len(local_test_ds)))

      in_sample_forecasts_acc = np.zeros((len(in_sample_forecasts), prediction_length))
      in_sample_actuals_acc = np.zeros((len(local_tss), prediction_length))

      for i in range(len(in_sample_forecasts)):
          in_sample_forecasts_acc[i] = np.mean(in_sample_forecasts[i].samples, axis=0)

      for i in range(len(in_sample_actuals_acc)):
          in_sample_actuals_acc[i] = local_tss[i][-(lookback_block+1)*prediction_length:-lookback_block*prediction_length].values.reshape(prediction_length)

      residuals = in_sample_actuals_acc - in_sample_forecasts_acc
      
      if lookback_block == 1:
        all_residuals = residuals
      else:
        all_residuals = np.hstack((residuals,all_residuals))


# ### Transform residuals to dataframe and save csv 
# 

# In[ ]:


if CALC_RESIDUALS:
  columns = []
  for i in range(1,(all_residuals.shape[1]+1)):
      columns.append("insample_"+str(i))
  all_res_df = pd.DataFrame(data=all_residuals, columns=columns)

  all_res_df = pd.concat([all_aggregates['id_str'],all_res_df],axis=1)


# In[ ]:


if CALC_RESIDUALS:
  all_res_df


# In[ ]:


if CALC_RESIDUALS:
  all_res_df.to_csv('{}/nbeats_toplvl_residuals_v{}.csv'.format(m5_input_path, VERSION), index=False)


# In[ ]:





# In[ ]:


if CALC_RESIDUALS:
  all_residuals.shape


# In[ ]:


if CALC_RESIDUALS:
  last_mean_residual = np.mean(residuals,axis=1)
  last_mean_residual


# In[ ]:


if CALC_RESIDUALS:
  mean_residual = np.mean(all_residuals,axis=1)
  mean_residual


# In[ ]:


if CALC_RESIDUALS:  
  local_tss[0][-(lookback_block+1)*single_prediction_length:-lookback_block*single_prediction_length].values.reshape(single_prediction_length)


# In[ ]:


if CALC_RESIDUALS:
  residuals[0]


# In[ ]:


if CALC_RESIDUALS:
  in_sample_actuals_acc[0]


# In[ ]:


if CALC_RESIDUALS:
  in_sample_forecasts_acc[0]


# # Predict and save forecast
# 

# In[ ]:


#calendar.tail(60)


# In[ ]:


forecasts[0]


# In[ ]:


forecasts_acc = np.zeros((len(forecasts), prediction_length))

for i in range(len(forecasts)):
    forecasts_acc[i] = forecasts[i].samples

columns = []
for i in range(1,(forecasts_acc.shape[1]+1)):
    columns.append("F"+str(i))
forecasts_acc_df = pd.DataFrame(data=forecasts_acc, columns=columns)


# In[ ]:


forecasts_acc_df = pd.concat([all_aggregates['id_str'],forecasts_acc_df],axis=1)


# In[ ]:


forecasts_acc_df


# In[ ]:


forecasts_acc_df.to_csv('{}/nbeats_toplvl_forecasts{}.csv'.format(m5_input_path, VERSION), index=False)

