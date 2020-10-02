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


####################################################################################
################## 2-3. nonrecursive model by store & dept #########################
####################################################################################


# In[ ]:


cvs = ['private']


# In[ ]:


STORES = ['CA_1', 'CA_2', 'CA_3', 'CA_4', 'TX_1', 'TX_2', 'TX_3', 'WI_1', 'WI_2', 'WI_3']
DEPTS = ['HOBBIES_1', 'HOBBIES_2', 'HOUSEHOLD_1', 'HOUSEHOLD_2', 'FOODS_1', 'FOODS_2', 'FOODS_3']


# In[ ]:


from  datetime import datetime, timedelta
import gc
import numpy as np, pandas as pd
import lightgbm as lgb

import os, sys, gc, time, warnings, pickle, psutil, random

warnings.filterwarnings('ignore')


# In[ ]:


def reduce_mem_usage(df, verbose=False):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                       df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# In[ ]:





# In[ ]:


FIRST_DAY = 710 
remove_feature = ['id',
                  'state_id',
                  'store_id',
#                   'item_id',
                  'dept_id',
                  'cat_id',
                  'date','wm_yr_wk','d','sales']

cat_var = ['item_id', 'dept_id','store_id', 'cat_id', 'state_id'] + ["event_name_1", "event_name_2", "event_type_1", "event_type_2"]
cat_var = list(set(cat_var) - set(remove_feature))


# In[ ]:


grid2_colnm = ['sell_price', 'price_max', 'price_min', 'price_std',
               'price_mean', 'price_norm', 'price_nunique', 'item_nunique',
               'price_momentum', 'price_momentum_m', 'price_momentum_y']

grid3_colnm = ['event_name_1', 'event_type_1', 'event_name_2',
               'event_type_2', 'snap_CA', 'snap_TX', 'snap_WI', 'tm_d', 'tm_w', 'tm_m',
               'tm_y', 'tm_wm', 'tm_dw', 'tm_w_end']

lag_colnm = [ 'sales_lag_28', 'sales_lag_29', 'sales_lag_30',
             'sales_lag_31', 'sales_lag_32', 'sales_lag_33', 'sales_lag_34',
             'sales_lag_35', 'sales_lag_36', 'sales_lag_37', 'sales_lag_38',
             'sales_lag_39', 'sales_lag_40', 'sales_lag_41', 'sales_lag_42',
             
             'rolling_mean_7', 'rolling_std_7', 'rolling_mean_14', 'rolling_std_14',
             'rolling_mean_30', 'rolling_std_30', 'rolling_mean_60',
             'rolling_std_60', 'rolling_mean_180', 'rolling_std_180']

mean_enc_colnm = [
    
    'enc_item_id_store_id_mean', 'enc_item_id_store_id_std'

]


# In[ ]:


########################### Make grid
#################################################################################
def prepare_data(store, state):
    
    grid_1 = pd.read_pickle(processed_data_dir+"grid_part_1.pkl")
    grid_2 = pd.read_pickle(processed_data_dir+"grid_part_2.pkl")[grid2_colnm]
    grid_3 = pd.read_pickle(processed_data_dir+"grid_part_3.pkl")[grid3_colnm]

    grid_df = pd.concat([grid_1, grid_2, grid_3], axis=1)
    del grid_1, grid_2, grid_3; gc.collect()
    
    grid_df = grid_df[(grid_df['store_id'] == store) & (grid_df['dept_id'] == state)]
    grid_df = grid_df[grid_df['d'] >= FIRST_DAY]
    
    lag = pd.read_pickle(processed_data_dir+"lags_df_28.pkl")[lag_colnm]
    
    lag = lag[lag.index.isin(grid_df.index)]
    
    grid_df = pd.concat([grid_df,
                     lag],
                    axis=1)
    
    del lag; gc.collect()
    

    mean_enc = pd.read_pickle(processed_data_dir+"mean_encoding_df.pkl")[mean_enc_colnm]
    mean_enc = mean_enc[mean_enc.index.isin(grid_df.index)]
    
    grid_df = pd.concat([grid_df,
                         mean_enc],
                        axis=1)    
    del mean_enc; gc.collect()
    
    grid_df = reduce_mem_usage(grid_df)
    
    
    
    return grid_df


# In[ ]:


validation = {
    'cv1' : [1551, 1610],
    'cv2' : [1829,1857],
    'cv3' : [1857, 1885],
    'cv4' : [1885,1913],
    'public' : [1913, 1941],
    'private' : [1941, 1969]
}


# ### cv1 : 2015-04-28 ~ 2015-06-26
# 
# ### cv2 : 2016-02-01 ~ 2016-02-28
# 
# ### cv3 : 2016-02-29 ~ 2016-03-27
# 
# ### cv4 : 2016-03-28 ~ 2016-04-24

# In[ ]:


########################### Model params
#################################################################################
lgb_params = {
                    'boosting_type': 'gbdt',
                    'objective': 'tweedie',
                    'tweedie_variance_power': 1.1,
                    'metric': 'rmse',
                    'subsample': 0.5,
                    'subsample_freq': 1,
                    'learning_rate': 0.015,
                    'num_leaves': 2**8-1,
                    'min_data_in_leaf': 2**8-1,
                    'feature_fraction': 0.5,
                    'max_bin': 100,
                    'n_estimators': 3000,
                    'boost_from_average': False,
                    'verbose': -1,
                    'seed' : 1995
                } 


# In[ ]:


########################### Predict 
#################################################################################

rmsse_bycv = dict()

for cv in cvs:
    print('cv : day', validation[cv])
    
    pred_list = []
    for store in STORES:
        for state in DEPTS:

            print(store,state, 'start')
            grid_df = prepare_data(store, state)

            model_var = grid_df.columns[~grid_df.columns.isin(remove_feature)]

            tr_mask = (grid_df['d'] <= validation[cv][0]) & (grid_df['d'] >= FIRST_DAY)
            vl_mask = (grid_df['d'] > validation[cv][0]) & (grid_df['d'] <= validation[cv][1])

            train_data = lgb.Dataset(grid_df[tr_mask][model_var], 
                           label=grid_df[tr_mask]['sales'])

            valid_data = lgb.Dataset(grid_df[vl_mask][model_var], 
                               label=grid_df[vl_mask]['sales'])
            
            model_path = model_dir+'non_recur_model_'+store+'_'+state+'.bin'
            m_lgb = pickle.load(open(model_path, 'rb'))
            
            
            indice = grid_df[vl_mask].index.tolist()
            prediction = pd.DataFrame({'y_pred' : m_lgb.predict(grid_df[vl_mask][model_var])})
            prediction.index = indice


            del grid_df, train_data, valid_data, m_lgb, tr_mask, vl_mask; gc.collect

            grid_1 = pd.read_pickle(processed_data_dir+"grid_part_1.pkl")
            pd.concat([grid_1.iloc[indice], prediction], axis=1)            .pivot(index='id', columns='d', values='y_pred')            .reset_index()            .set_index('id')            .to_csv(log_dir+f'submission_storeanddept_{store}_{state}_{cv}.csv')



            del grid_1; gc.collect()


# In[ ]:


########################### Make submissions
#################################################################################

os.chdir(log_dir)

pri = [a for a in os.listdir() if 'storeanddept' in a]

os.chdir(dir_)

submission = pd.read_csv(raw_data_dir+'sample_submission.csv').set_index('id').iloc[30490:]
sub_id = pd.DataFrame({'id':submission.index.tolist()})

fcol = [f'F{i}' for i in range(1,29)]

sub_copy = submission.copy()
for file in pri:
    temp = pd.read_csv(log_dir+file)
    temp.columns = ['id']+fcol
    sub_copy += sub_id.merge(temp, how='left', on='id').set_index('id').fillna(0)
sub_copy.columns = fcol
sub_copy.to_csv(submission_dir+'before_ensemble/submission_kaggle_nonrecursive_store_dept.csv')

