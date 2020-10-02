#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

pd.options.display.max_columns = 50

import random
from  datetime import datetime, timedelta

import lightgbm as lgb


# In[2]:


CAL_DTYPES={"event_name_1": "category", "event_name_2": "category", "event_type_1": "category", 
         "event_type_2": "category", "weekday": "category", 'wm_yr_wk': 'int16', "wday": "int16",
        "month": "int16", "year": "int16", "snap_CA": "float32", 'snap_TX': 'float32', 'snap_WI': 'float32' }
PRICE_DTYPES = {"store_id": "category", "item_id": "category", "wm_yr_wk": "int16","sell_price":"float32", 
                "rm_diff_price_4":"float32", "rm_diff_price_12":"float32","rm_diff_price_50":"float32" }
PROC_PRICES_DTYPES = {"store_id": "category", "item_id": "category", "wm_yr_wk": "int16", 
                        "rm_diff_price_4":"float32", "rm_diff_price_12":"float32","rm_diff_price_50":"float32" }


# In[3]:


h = 28 
max_lags = 70
tr_last = 1941
fday = datetime(2016,5, 23) 
fday


# In[4]:


def create_dt(is_train = True, nrows = None, first_day = 1200, dept='HOBBIES_1'):
    prices = pd.read_csv("./raw_data/sell_prices.csv", dtype = PRICE_DTYPES)
    proc_price = pd.read_csv('./proc_data/prices_processed.csv', dtype = PROC_PRICES_DTYPES).drop('sell_price', axis=1)
    prices = prices.merge(proc_price, on=['store_id','item_id','wm_yr_wk'], how='left')
    del(proc_price)
    
    for col, col_dtype in PRICE_DTYPES.items():
        if col_dtype == "category":
            prices[col] = prices[col].cat.codes.astype("int16")
            prices[col] -= prices[col].min()
            
    cal = pd.read_csv("./raw_data/calendar.csv", dtype = CAL_DTYPES)
    proc_cal = pd.read_csv('./proc_data/processed_calendar.csv').drop(
    ['wm_yr_wk','wday','month','year','snap_CA','snap_TX','snap_WI'], axis=1).rename(columns={'day':'d'})
    cols_events_days = ['d_{}'.format(c) for c in list(np.arange(1910,1990))]
    ev1 = cal[cal['d'].isin(cols_events_days)]['event_name_1'].unique().tolist()
    ev2 = cal[cal['d'].isin(cols_events_days)]['event_name_2'].unique().tolist()
    evs = list(set(ev1+ev2))
    for c in list(set(proc_cal.columns.tolist()) - set(['d'])):
        proc_cal[c] = proc_cal[c].astype(int)
    cal = cal.merge(proc_cal, on='d', how='left')
    cal["date"] = pd.to_datetime(cal["date"])
    for col, col_dtype in CAL_DTYPES.items():
        if col_dtype == "category":
            cal[col] = cal[col].cat.codes.astype("int16")
            cal[col] -= cal[col].min()
    events_to_mantain = ['event_name_1_{}'.format(c) for c in evs]+['event_name_2_{}'.format(c) for c in evs]
    events_to_mantain_ = [c for c in cal.columns if c in events_to_mantain]
    cal = cal[['date','wm_yr_wk','weekday','wday','month','year','d','event_name_1','event_type_1','event_name_2',
               'event_type_2','snap_CA','snap_TX','snap_WI','group_day']+events_to_mantain_]
    
    start_day = max(1 if is_train  else tr_last-max_lags, first_day)
    numcols = [f"d_{day}" for day in range(start_day,tr_last+1)]
    catcols = ['id', 'item_id', 'dept_id','store_id', 'cat_id', 'state_id']
    dtype = {numcol:"float32" for numcol in numcols} 
    dtype.update({col: "category" for col in catcols if col != "id"})
    dt = pd.read_csv("./raw_data/sales_train_evaluation.csv", 
                     nrows = nrows, usecols = catcols + numcols, dtype = dtype)
    dt = dt[dt['dept_id']==dept]
    
    for col in catcols:
        if col != "id":
            dt[col] = dt[col].cat.codes.astype("int16")
            dt[col] -= dt[col].min()
    
    if not is_train:
        for day in range(tr_last+1, tr_last+ 28 +1):
            dt[f"d_{day}"] = np.nan
    
    dt = pd.melt(dt,
                  id_vars = catcols,
                  value_vars = [col for col in dt.columns if col.startswith("d_")],
                  var_name = "d",
                  value_name = "sales")
    
    dt = dt.merge(cal, on= "d", copy = False)
    dt = dt.merge(prices, on = ["store_id", "item_id", "wm_yr_wk"], copy = False)
    
    return dt


# In[5]:


def create_fea(dt):
    lags = [7, 28]
    lag_cols = [f"lag_{lag}" for lag in lags ]
    for lag, lag_col in zip(lags, lag_cols):
        dt[lag_col] = dt[["id","sales"]].groupby("id")["sales"].shift(lag)

    wins = [7, 28]
    for win in wins :
        for lag,lag_col in zip(lags, lag_cols):
            dt[f"rmean_{lag}_{win}"] = dt[["id", lag_col]].groupby("id")[lag_col].transform(lambda x : x.rolling(win).mean())

    
    
    date_features = {
        
        "wday": "weekday",
        "week": "weekofyear",
        "month": "month",
        "quarter": "quarter",
        "year": "year",
        "mday": "day",
#         "ime": "is_month_end",
#         "ims": "is_month_start",
    }
    
#     dt.drop(["d", "wm_yr_wk", "weekday"], axis=1, inplace = True)
    
    for date_feat_name, date_feat_func in date_features.items():
        if date_feat_name in dt.columns:
            dt[date_feat_name] = dt[date_feat_name].astype("int16")
        else:
            dt[date_feat_name] = getattr(dt["date"].dt, date_feat_func).astype("int16")


# In[6]:


FIRST_DAY = 350


# In[7]:


get_ipython().run_cell_magic('time', '', '\nsub_p_total = pd.DataFrame()\nfor dept in [\'HOBBIES_1\', \'HOBBIES_2\', \'HOUSEHOLD_1\', \'HOUSEHOLD_2\', \'FOODS_1\', \'FOODS_2\', \'FOODS_3\']:\n#for dept in [\'HOBBIES_1\']:\n\n    df = create_dt(is_train=True, first_day= FIRST_DAY, dept=dept)\n    print(df.shape)\n\n    create_fea(df)\n    print(df.shape)\n    print(df.columns)\n\n    for c in [c for c in df.columns.tolist() if \'rm_diff_price_\' in c]:\n        df[c].fillna(0, inplace=True)\n    \n    #df.dropna(inplace = True)\n    df.shape\n\n    cat_feats = ([\'item_id\',\'store_id\', \'cat_id\', \'state_id\'] \n                 + ["event_type_1", "event_type_2"]\n                 + [\'wday\', \'month\', \'snap_CA\', \'snap_TX\', \'snap_WI\'])\n    \n    useless_cols = ["id", "date", "sales","d", "wm_yr_wk", "weekday", "dept_id", "sell_price",\'event_name_1\', \'event_name_2\']\n    train_cols = df.columns[~df.columns.isin(useless_cols)]\n\n    #days_val = df[\'d\'].unique().tolist()[-200:]\n    days_val = random.choices(df[\'d\'].unique().tolist(), k=500)\n    X_train = df[df[\'d\'].isin(days_val)==False][train_cols]\n    y_train = df[df[\'d\'].isin(days_val)==False]["sales"]\n    X_val = df[df[\'d\'].isin(days_val)==True][train_cols]\n    y_val = df[df[\'d\'].isin(days_val)==True]["sales"]\n\n    train_data = lgb.Dataset(X_train, label = y_train, categorical_feature=cat_feats)\n    valid_data = lgb.Dataset(X_val, label = y_val, categorical_feature=cat_feats)\n\n    params = {\n            "objective" : "poisson",\n            "metric" :"poisson",\n            "learning_rate" : 0.09,\n            "sub_feature" : 0.9,\n            "sub_row" : 0.75,\n            "bagging_freq" : 1,\n            "lambda_l2" : 0.1,\n            \'verbosity\': 1,\n            \'num_iterations\' : 2000,\n            \'num_leaves\': 32,\n            "min_data_in_leaf": 50,\n    }\n\n    m_lgb = lgb.train(params, train_data, valid_sets = [train_data, valid_data], \n                      verbose_eval=20, early_stopping_rounds=30) \n    \n    feature_imp = pd.DataFrame({\'Value\':m_lgb.feature_importance(),\'Feature\':X_train.columns})\n    feature_imp = feature_imp.sort_values(by=\'Value\', ascending=False).reset_index(drop=True)\n    \n    display(feature_imp.head(20))\n\n    te = create_dt(False, dept=dept)\n    cols = [f"F{i}" for i in range(1,29)]\n\n    for tdelta in range(0, 28):\n        day = fday + timedelta(days=tdelta)\n        print(day)\n        tst = te[(te.date >= day - timedelta(days=max_lags)) & (te.date <= day)].copy()\n        create_fea(tst)\n        tst = tst.loc[tst.date == day , train_cols]\n        te.loc[te.date == day, "sales"] = m_lgb.predict(tst)\n        del(tst)\n\n    del(m_lgb)\n    sub_p = pd.pivot_table(te, index=\'id\', values=\'sales\', columns=\'d\').iloc[:,-28:].reset_index()\n    del(te)\n    sub_p_total = pd.concat([sub_p_total, sub_p])\n    del(sub_p)')


# In[8]:


sub_p_total.shape


# In[9]:


sub_p_total.tail()


# In[10]:


sub = pd.read_csv('./raw_data/sample_submission.csv', usecols=['id'])


# In[11]:


sub = sub.merge(sub_p_total, on='id', how='left')


# In[12]:


sub = sub.dropna()


# In[13]:


sub2 = sub.copy()
sub2["id"] = sub2["id"].str.replace("validation$", "evaluation")
sub = pd.concat([sub, sub2], axis=0, sort=False)
sub.columns = ['id'] + ['F' + str(c) for c in np.arange(1,29,1)]
sub.to_csv("./proc_data/partial_submission.csv",index=False)

