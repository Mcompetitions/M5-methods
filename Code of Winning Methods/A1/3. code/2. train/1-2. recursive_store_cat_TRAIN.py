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


# In[ ]:


####################################################################################
####################### 1-2. recursive model by store & cat ########################
####################################################################################


# In[ ]:


ver, KKK = 'priv', 0


# In[ ]:


STORES = ['CA_1', 'CA_2', 'CA_3', 'CA_4', 'TX_1', 'TX_2', 'TX_3', 'WI_1', 'WI_2', 'WI_3']
CATS = ['HOBBIES','HOUSEHOLD', 'FOODS']


# In[ ]:


# General imports
import numpy as np
import pandas as pd
import os, sys, gc, time, warnings, pickle, psutil, random

# custom imports
from multiprocessing import Pool

warnings.filterwarnings('ignore')


# In[ ]:


########################### Helpers
#################################################################################
def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)

    
## Multiprocess Runs
def df_parallelize_run(func, t_split):
    num_cores = np.min([N_CORES,len(t_split)])
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, t_split), axis=1)
    pool.close()
    pool.join()
    return df


# In[ ]:


########################### Helper to load data by store ID
#################################################################################
# Read data
def get_data_by_store(store, dept):
    
    df = pd.concat([pd.read_pickle(BASE),
                    pd.read_pickle(PRICE).iloc[:,2:],
                    pd.read_pickle(CALENDAR).iloc[:,2:]],
                    axis=1)
    
    df = df[df['d']>=START_TRAIN]
    

    df = df[(df['store_id']==store) & (df['cat_id']==dept)]

    df2 = pd.read_pickle(MEAN_ENC)[mean_features]
    df2 = df2[df2.index.isin(df.index)]
        
    df3 = pd.read_pickle(LAGS).iloc[:,3:]
    df3 = df3[df3.index.isin(df.index)]
    
    df = pd.concat([df, df2], axis=1)
    del df2
    
    df = pd.concat([df, df3], axis=1)
    del df3
    
    features = [col for col in list(df) if col not in remove_features]
    df = df[['id','d',TARGET]+features]
    
    df = df.reset_index(drop=True)
    
    return df, features

# Recombine Test set after training
def get_base_test():
    base_test = pd.DataFrame()

    for store_id in STORES:
        for state_id in CATS:
            temp_df = pd.read_pickle(processed_data_dir+'test_'+store_id+'_'+state_id+'.pkl')
            temp_df['store_id'] = store_id
            temp_df['cat_id'] = state_id
            base_test = pd.concat([base_test, temp_df]).reset_index(drop=True)
    
    return base_test


########################### Helper to make dynamic rolling lags
#################################################################################
def make_lag(LAG_DAY):
    lag_df = base_test[['id','d',TARGET]]
    col_name = 'sales_lag_'+str(LAG_DAY)
    lag_df[col_name] = lag_df.groupby(['id'])[TARGET].transform(lambda x: x.shift(LAG_DAY)).astype(np.float16)
    return lag_df[[col_name]]


def make_lag_roll(LAG_DAY):
    shift_day = LAG_DAY[0]
    roll_wind = LAG_DAY[1]
    lag_df = base_test[['id','d',TARGET]]
    col_name = 'rolling_mean_tmp_'+str(shift_day)+'_'+str(roll_wind)
    lag_df[col_name] = lag_df.groupby(['id'])[TARGET].transform(lambda x: x.shift(shift_day).rolling(roll_wind).mean())
    return lag_df[[col_name]]


# In[ ]:


########################### Model params
#################################################################################
import lightgbm as lgb
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
                    'verbose': -1
                } 


# In[ ]:





# In[ ]:


########################### Vars
#################################################################################
VER = 1                        
SEED = 42                      
seed_everything(SEED)
lgb_params['seed'] = SEED
N_CORES = psutil.cpu_count()


#LIMITS and const
TARGET      = 'sales'
START_TRAIN = 700 
END_TRAIN   = 1941 - 28*KKK
P_HORIZON   = 28        
USE_AUX     = False

remove_features = ['id','cat_id', 'state_id','store_id',
                   'date','wm_yr_wk','d',TARGET]
mean_features   = ['enc_store_id_dept_id_mean','enc_store_id_dept_id_std',
                   'enc_item_id_store_id_mean','enc_item_id_store_id_std'] 

ORIGINAL = raw_data_dir
BASE     = processed_data_dir+'grid_part_1.pkl'
PRICE    = processed_data_dir+'grid_part_2.pkl'
CALENDAR = processed_data_dir+'grid_part_3.pkl'
LAGS     = processed_data_dir+'lags_df_28.pkl'
MEAN_ENC = processed_data_dir+'mean_encoding_df.pkl'


SHIFT_DAY  = 28
N_LAGS     = 15
LAGS_SPLIT = [col for col in range(SHIFT_DAY,SHIFT_DAY+N_LAGS)]
ROLS_SPLIT = []
for i in [1,7,14]:
    for j in [7,14,30,60]:
        ROLS_SPLIT.append([i,j])


# In[ ]:


########################### Train Models
#################################################################################
for store_id in STORES:
    for state_id in CATS:
        print('Train', store_id, state_id)

        grid_df, features_columns = get_data_by_store(store_id, state_id)

        train_mask = grid_df['d']<=END_TRAIN
        valid_mask = train_mask&(grid_df['d']>(END_TRAIN-P_HORIZON))
        preds_mask = (grid_df['d']>(END_TRAIN-100)) & (grid_df['d'] <= END_TRAIN+P_HORIZON)

        train_data = lgb.Dataset(grid_df[train_mask][features_columns], 
                           label=grid_df[train_mask][TARGET])

        valid_data = lgb.Dataset(grid_df[valid_mask][features_columns], 
                           label=grid_df[valid_mask][TARGET])

        grid_df = grid_df[preds_mask].reset_index(drop=True)
        keep_cols = [col for col in list(grid_df) if '_tmp_' not in col]
        grid_df = grid_df[keep_cols]

        d_sales = grid_df[['d','sales']]
        substitute = d_sales['sales'].values
        substitute[(d_sales['d'] > END_TRAIN)] = np.nan
        grid_df['sales'] = substitute

        grid_df.to_pickle(processed_data_dir+'test_'+store_id+'_'+state_id+'.pkl')
        del grid_df, d_sales, substitute

        seed_everything(SEED)
        estimator = lgb.train(lgb_params,
                              train_data,
                              valid_sets = [valid_data],
                              verbose_eval = 100
                              
                              )
        
        # display(pd.DataFrame({'name':estimator.feature_name(),
        #                       'imp':estimator.feature_importance()}).sort_values('imp',ascending=False).head(25))
        
        model_name = model_dir+'lgb_model_'+store_id+'_'+state_id+'_v'+str(VER)+'.bin'
        pickle.dump(estimator, open(model_name, 'wb'))

        
        del train_data, valid_data, estimator
        gc.collect()

        MODEL_FEATURES = features_columns

