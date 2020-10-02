#!/usr/bin/env python
# coding: utf-8

# In[32]:


# Instructions:
#
# model will need to run separately in batches:
#
#    LEVEL = 13, MAX_LEVEL = None  (1 hour train)
#    LEVEL = 14, MAX_LEVEL = None     ibid.
#    LEVEL = 15, MAX_LEVEL = None     ibid.
#    LEVEL = -1, MAX_LEVEL = 11    (~10 hour train)

# you could probably set MAX_LEVEL = 15 and train/infer all at once if you had 4x16GB RAM 

# to predict from saved models:
#    use each of the above settings with IMPORT = True; (runtime <10 minutes per run)

# the FINAL_BASE parameter determines whether to forecast the evaluation or validation period

# the SPEED = True flag reduces runtimes by 20x and appears to deliver identical performance (0.0% dif in CV)
# you may replicate the original submission by setting SPEED = False (200 hours training, 10 hours inference)

# In[33]:
LEVEL = 15  # Level 13 is HOBBIES; Level 14 is HOUSEHOLD; Level 15 is FOODS (there is no "Level 12")
MAX_LEVEL = None
IMPORT = False 

FINAL_BASE = ['d_1941', 'd_1913'][0]


SINGLE_FOLD = True
SPEED = False
SUPER_SPEED = False
REDUCED_FEATURES = False

sparse_features = ['dayofweek', 'dayofmonth', 'qs_30d_ewm', 'qs_100d_ewm', 'qs_median_28d', 'qs_mean_28d', 'state_id',
                   'qs_qtile90_28d', 'pct_nonzero_days_28d', 'days_fwd']
LEVEL_SPLITS = [(13, 'HOBBIES'), (14, 'HOUSEHOLD'), (15, 'FOODS')  ]
# ID_FILTER = '';   #  ['HOBBIES', 'HOUSEHOLD', 'FOODS', ]


# In[34]:
QUANTILES = [0.005, 0.025, 0.165, 0.25, 0.5,  0.75, 0.835, 0.975, 0.995]
# QUANTILES = [0.25, 0.5, 0.75]
# QUANTILES = [0.5]


# In[35]:
P_DICT = {1: (0.3, 0.7), 2: (0.1, 0.7), 3: (0.1, 0.5), 4: (0.3, 0.5), 5: (0.15, 1), 6: (0.2, 0.5), 7: (0.1, 1),
          8: (0.2, 0.5), 9: (0.1, 0.5), 10: (0.05, 0.5), 11: (0.04, 1), 13: (0.12, 2), 14: (0.065, 2), 15: (0.03, 0.5)}
#     'HOBBIES': (0.12, 2), 'HOUSEHOLD': (0.065, 2), 'FOODS': (0.03, 0.5)}

SS_SS = 0.8    # 0.8 was production version ***

if SPEED or SUPER_SPEED or REDUCED_FEATURES:
    SS_SS /= (5 if SUPER_SPEED else (2 if SPEED else 1)) * (5 if REDUCED_FEATURES else 1)


# In[36]:
BAGS = 1
N_JOBS = -1

SS_PWR = 0.6
BAGS_PWR = 0


# In[37]:
# levels


# In[38]:
FEATURE_DROPS = ['item_id', '_abs_diff', 'squared_diff' ]                +    ['336', '300d']


# In[39]:
# run-time parameters
CACHED_FEATURES = False
CACHE_FEATURES = False


# In[40]:
TIME_SEED = True


# ### Load Packages and Settings
# In[41]:
# !pip install lightgbm


# In[42]:
import numpy as np
import pandas as pd 


# In[43]:
import psutil
import os


# In[44]:
import pickle


# In[45]:
from collections import Counter
import datetime as datetime
from scipy.stats.mstats import gmean
import random


# In[46]:
import gc
import gzip
import bz2


# In[47]:
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = (17,5.5)
rcParams['figure.max_open_warning'] = 0
# %config InlineBackend.figure_format='retina'


# In[48]:
import seaborn as sns


# In[49]:
pd.options.display.max_rows = 150


# In[50]:
start = datetime.datetime.now()


# In[51]:
if TIME_SEED:
    np.random.seed(datetime.datetime.now().microsecond)


# In[52]:
import sys


def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)


def memCheck():
    for name, size in sorted(((name, sys.getsizeof(value)) for name, value in globals().items()),
                             key= lambda x: -x[1])[:10]:
        print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))


# In[53]:
def ramCheck():
    print("{:.1f} GB used".format(psutil.virtual_memory().used/1e9 - 0.7))


# In[54]:
path = '../input/Data'


# In[55]:
# path = 'm5-data'


# In[56]:
ramCheck()


# ### Load and Aggregate Training Data
# In[57]:
LEVELS = [(12, ['item_id', 'store_id']),
          (11, ['state_id', 'item_id']),
          (10, ['item_id']),
          (9, ['store_id', 'dept_id']),
          (8, ['store_id', 'cat_id']),
          (7, ['state_id', 'dept_id']),
          (6, ['state_id', 'cat_id']),
          (5, ['dept_id']),
          (4, ['cat_id']),
          (3, ['store_id']),
          (2, ['state_id']),
          (1, []) ]

DOWNSTREAM = {'item_id': ['dept_id', 'cat_id'],
              'dept_id': ['cat_id'],
              'store_id': ['state_id']}


# In[58]:
def aggTrain(train):
    tcd = dict([(col, 'first') for col in train.columns[1:6]])
    tcd.update( dict([(col, 'sum') for col in train.columns[6:]]))

    tadds =[]; tadd_levels= [ [12 for i in range(0, len(train))] ] 
    for idx, lvl in enumerate(LEVELS[1:]):
        level = lvl[0]
        lvls = lvl[1]

        if len(lvls) is 0:  # group all if no list provided
            lvls = [1 for i in range(0, len(train))]

        tadd = train.groupby(lvls).agg(tcd)

        # name it
        if len(lvls) == 2:
            tadd.index = ['_'.join(map(str,i)) for i in tadd.index.tolist()]
        elif len(lvls) == 1:
            tadd.index = tadd.index + '_X'
        else:
            tadd.index = ['Total_X']
        tadd.index.name = 'id'

        # fill in categorical features
        tadd.reset_index(inplace=True)
        for col in [c for c in train.columns[1:6] if c not in lvls and not  
                            any(c in z for z in[DOWNSTREAM[lvl] for lvl in lvls if lvl in DOWNSTREAM])]:
            tadd[col] = 'All'
        tadds.append(tadd)

        #levels
        tadd_levels.append([level for i in range(0, len(tadd))])

    train = pd.concat((train,*tadds), sort=False, ignore_index=True); del tadds, tadd
    levels = pd.Series(data = [x for sub_list in tadd_levels for x in sub_list], index = train.index); del tadd_levels
    for col in train.columns[1:6]:
        train[col] = train[col].astype('category')
        
    return train, levels


# In[59]:
def loadTrain():
    train_cols = pd.read_csv(path + '/' + 'sales_train_evaluation.csv', nrows=1)

    c_dict = {}
    for col in [c for c in train_cols if 'd_' in c]:
        c_dict[col] = np.float64

    train = pd.read_csv(path + '/' + 'sales_train_evaluation.csv', dtype=c_dict)  # .astype(np.int16, errors='ignore')

    train.id = train.id.str.split('_').str[:-1].str.join('_')
    
    train.sort_values('id', inplace=True)
    
    return train.reset_index(drop=True)


# In[60]:
def getPricePivot():
    prices = pd.read_csv(path+ '/' + 'sell_prices.csv',
                    dtype = {'wm_yr_wk': np.int16, 'sell_price': np.float64})
    prices['id'] = prices.item_id + "_" + prices.store_id
    price_pivot = prices.pivot(columns = 'id' , index='wm_yr_wk', values = 'sell_price')
    price_pivot = price_pivot.reindex(sorted(price_pivot.columns), axis=1)
    return price_pivot


# In[61]:
def getCal():
    return pd.read_csv(path+ '/' + 'calendar.csv').set_index('d')


# In[62]:
cal = getCal()
cal.date = pd.to_datetime(cal.date)

day_to_cal_index = dict([(col, idx) for idx, col in enumerate(cal.index)])
cal_index_to_day = dict([(idx, col) for idx, col in enumerate(cal.index)])

cal_index_to_wm_yr_wk = dict([(idx, col) for idx, col in enumerate(cal.wm_yr_wk)])
day_to_wm_yr_wk = dict([(idx, col) for idx, col in cal.wm_yr_wk.iteritems()])


# In[63]:
# Load
train = loadTrain()
price_pivot = getPricePivot()


# In[64]:
print('Total Time Elapsed: ', (datetime.datetime.now() - start).seconds, 's')


# In[65]:
# combine
assert (train.id == price_pivot.columns).all()
daily_sales = pd.concat((train.iloc[:, :6], 
                        train.iloc[:, 6:] * price_pivot.loc[train.columns[6:].fillna(0)\
                                                                .map(day_to_wm_yr_wk)].transpose().values ), 
                            axis = 'columns')


# In[66]:
# Aggregate
train, levels = aggTrain(train)
# id_to_level = dict(zip(train.id, levels))
# level_to_ids = dict([(level[0], list(train.id[levels == level[0]])) for idx, level in enumerate(LEVELS)])

daily_sales = aggTrain(daily_sales)[0]


# In[67]:
print('Total Time Elapsed: ', (datetime.datetime.now() - start).seconds, 's')


# In[68]:
# Rescale each level to avoid hitting np.float64 ceiling and keep similar ranges
level_multiplier = dict([ (c, (levels==c).sum() / (levels==12).sum()) for c in sorted(levels.unique())])


# In[69]:
# split up level 12
for row in LEVEL_SPLITS:
    level_multiplier[row[0]] = level_multiplier[12]
    levels.loc[(levels == 12) & (train.cat_id == row[1])] = row[0]


# In[70]:
Counter(levels)


# In[71]:
# Rescale by number of series at each level
train = pd.concat((train.iloc[:, :6], 
                        train.iloc[:, 6:].multiply( levels.map(level_multiplier), axis = 'index').astype(np.float64) ), 
                            axis = 'columns')

daily_sales = pd.concat((daily_sales.iloc[:, :6], 
                        daily_sales.iloc[:, 6:].multiply( levels.map(level_multiplier), axis = 'index').astype(np.float64) ), 
                            axis = 'columns')


# In[72]:


def loadSampleSub():
    return pd.read_csv(path+ '/' + 'sample_submission.csv').astype(np.int8, errors = 'ignore')

sample_sub = loadSampleSub()

assert set(train.id) == set(sample_sub.id.str.split('_').str[:-2].str.join('_'))


# In[73]:


print(len(train))


# In[ ]:





# In[74]:


ramCheck()


# In[75]:


# memCheck()


# In[76]:


print('Total Time Elapsed: ', (datetime.datetime.now() - start).seconds, 's')


# In[ ]:





# In[77]:


train_filter = (   
               ( ( MAX_LEVEL is not None )   & (levels <= MAX_LEVEL) )  | 
               (  ( MAX_LEVEL is None )  &  (levels == LEVEL) )
                 )
train = train[train_filter].reset_index(drop=True)
daily_sales = daily_sales[train_filter].reset_index(drop=True)
levels = levels[train_filter].reset_index(drop=True).astype(np.int8)


# In[78]:


Counter(levels)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[79]:


train.head()


# In[80]:


print(len(train))


# In[ ]:





# In[81]:


train_head = train.iloc[:, :6]  


# In[82]:


train_head.head()


# In[83]:


ramCheck()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[84]:


# replace leading zeros with nan
train['d_1'].replace(0, np.nan, inplace=True)

for i in range(train.columns.get_loc('d_1') + 1, train.shape[1]):
    train.loc[:, train.columns[i]].where( ~ ((train.iloc[:,i]==0) & (train.iloc[:,i-1].isnull())),
                                         np.nan, inplace=True)


# In[85]:


print('Total Time Elapsed: ', (datetime.datetime.now() - start).seconds, 's')


# In[86]:


train.head(5)


# In[ ]:





# In[ ]:





# In[87]:


train_flipped = train.set_index('id', drop = True).iloc[:, 5:].transpose()


# In[88]:


train_flipped.dtypes


# In[89]:


train_flipped.head()


# In[90]:


train_flipped.max().sort_values(ascending=False)[::3000]


# In[ ]:





# In[91]:


# memCheck()


# In[92]:


print('Total Time Elapsed: ', (datetime.datetime.now() - start).seconds, 's')


# In[93]:


ramCheck()


# In[ ]:





# In[ ]:





# In[94]:


print('Total Time Elapsed: ', (datetime.datetime.now() - start).seconds, 's')


# In[ ]:





# ### Item-Store Features

# In[95]:


features = []


# In[96]:


# basic moving averages
if not CACHED_FEATURES:      
    for window in [3, 7, 15, 30, 100]:
        if REDUCED_FEATURES and window < 15: continue;
        features.append(('qs_{}d_ewm'.format(window), 
                         train_flipped.ewm(span=window, 
                                           min_periods = int(np.ceil(window ** 0.8))  ).mean().astype(np.float64)))
 


# In[97]:


store_avg_qs = train_flipped[train_flipped.columns[levels >= 12]].transpose()            .groupby(train_head.iloc[(levels >= 12).values].store_id.values).mean().fillna(1)
store_dept_avg_qs = train_flipped[train_flipped.columns[levels >= 12]].transpose()            .groupby(  ( train_head.iloc[(levels >= 12).values].store_id.astype(str) + '_'
                        + train_head.iloc[(levels >= 12).values].dept_id.astype(str)).values
                    ).mean().fillna(1)


# In[98]:


store_avg_qs


# In[99]:


# basic moving averages, after removing any store trends
scaled_sales = train_flipped / (store_avg_qs.loc[train.store_id].transpose().values); 

# if levels.min() == 12:
#     # get overall store and store-dept sales matched to this id;
#     store_avg_qs_matched = store_avg_qs.loc[train.store_id].transpose() 
#     store_dept_avg_qs_matched = store_dept_avg_qs.loc[train.store_id.astype(str) + '_'
#                                                   + train.dept_id.astype(str)
#                                                 ].transpose() 

#     store_avg_qs_matched.columns = train_flipped.columns
#     store_dept_avg_qs_matched.columns = train_flipped.columns

#     ratio = (store_avg_qs_matched.rolling(28).mean() / store_avg_qs_matched.rolling(56).mean() ) .fillna(1) - 1
#     ratio = ratio.clip ( ratio.stack().quantile(0.01), ratio.stack().quantile(0.99))
# #     features.append(('store_28d_58d_ratio',  ratio.astype(np.float64)))

#     ratio = (store_dept_avg_qs_matched.rolling(28).mean() / store_dept_avg_qs_matched.rolling(56).mean() ) .fillna(1) - 1
#     ratio = ratio.clip ( ratio.stack().quantile(0.003), ratio.stack().quantile(0.997))

# #     features.append(('store_dept_28d_58d_ratio',  ratio.astype(np.float64)))

#     del store_avg_qs_matched, store_dept_avg_qs_matched, ratio

del store_avg_qs, store_dept_avg_qs,


# In[ ]:





# In[ ]:





# In[100]:


# moving average after store-level detrending
if not CACHED_FEATURES:
    for window in [3, 7, 15, 30, 100]:
        if REDUCED_FEATURES: continue;
        features.append(('qs_divbystore_{}d_ewm'.format(window), 
                         scaled_sales.ewm(span=window,
                                           min_periods = int(np.ceil(window ** 0.8))  ).mean().astype(np.float64)))


# In[101]:


print('Total Time Elapsed: ', (datetime.datetime.now() - start).seconds, 's')


# In[ ]:





# In[102]:


# EWM % NONZERO DAYS
if not CACHED_FEATURES:
    tff0ne0 = train_flipped.fillna(0).ne(0)
    for window in [7, 14, 28, 28*2, 28*4,  ]:  
        if REDUCED_FEATURES and window != 28: continue;
        features.append( ('pct_nonzero_days_{}d'.format(window),
                         tff0ne0.rolling(window).mean().astype(np.float64) ) )
    del tff0ne0


# In[ ]:





# In[103]:


print('Total Time Elapsed: ', (datetime.datetime.now() - start).seconds, 's')


# In[ ]:





# ### Features for Both Sales and Scaled Sales

# In[104]:


arrs = [train_flipped, scaled_sales, ] # sales_over_all]
labels = ['qs', 'qs_divbystore', ] #'qs_divbyall']

if REDUCED_FEATURES: arrs = arrs[0:1]


# In[105]:


# basic lag features
if not CACHED_FEATURES:
    for lag in range(1, 10+1):
        if REDUCED_FEATURES: continue;
        features.append( ('qs_lag_{}d'.format(lag),
                              train_flipped.shift(lag).fillna(0).astype(np.float64) ) )


# In[106]:


# means and medians -- by week to avoid day of week effects

if not CACHED_FEATURES:
    for idx in range(0, len(arrs)):
        arr = arrs[idx]
        label = labels[idx]

        for window in [7, 14, 21, 28, 28*2, 28*4,  ]:  ## ** mean and median
            if REDUCED_FEATURES and window != 28: continue;
            features.append( ('{}_mean_{}d'.format(label, window), 
                          arr.rolling(window).mean().astype(np.float64) )  )

            features.append( ('{}_median_{}d'.format(label, window), 
                          arr.rolling(window).median().astype(np.float64) )  )
            
            print('{}: {}'.format(label,window))

        del arr


# In[107]:


print('Total Time Elapsed: ', (datetime.datetime.now() - start).seconds, 's')


# In[ ]:





# In[108]:


# stdev, skewness, and kurtosis
# ideally kurtosis and skewness should NOT be labeled qs_ as they are scale-invariant

if not CACHED_FEATURES:
    for idx in range(0, len(arrs)):
        arr = arrs[idx]
        label = labels[idx]
        for window in [7, 14, 28, 28*3, 28*6]:
            if REDUCED_FEATURES and window != 28: continue;
            print('{}: {}'.format(label,window))

            features.append( ('{}_stdev_{}d'.format(label, window), 
                                  arr.rolling(window).std().astype(np.float64) )  )

            if window >= 10:
                if REDUCED_FEATURES: continue;
                features.append( ('{}_skew_{}d'.format(label, window), 
                                      arr.rolling(window).skew().astype(np.float64) )  )

                features.append( ('{}_kurt_{}d'.format(label, window), 
                                      arr.rolling(window).kurt().astype(np.float64) )  )

    del arr;


# In[109]:


print('Total Time Elapsed: ', (datetime.datetime.now() - start).seconds, 's')


# In[ ]:





# In[110]:


# high and low quantiles (adding more seemed to hurt performance)

if not CACHED_FEATURES:
    for idx in range(0, len(arrs)):
        arr = arrs[idx]
        label = labels[idx]
        for window in [14, 28, 56]:
            if REDUCED_FEATURES and window != 28: continue;

            features.append( ('{}_qtile10_{}d'.format(label, window), 
                          arr.rolling(window).quantile(0.1).astype(np.float64) )  )

            features.append( ('{}_qtile90_{}d'.format(label, window), 
                          arr.rolling(window).quantile(0.9).astype(np.float64) )  )

            print('{}: {}'.format(label,window))
        del arr


# In[111]:


print('Total Time Elapsed: ', (datetime.datetime.now() - start).seconds, 's')


# In[ ]:





# In[112]:


del arrs; del scaled_sales


# In[113]:


ramCheck()


# In[ ]:





# ### Data Cleaning

# In[114]:


# start after one year, remove anything with proximity to holiday months (given mid-year LB targets)
# also saves a lot of RAM/processing time 

def clean_df(fr):
    early_rows = cal[cal.year == cal.year.min()].index.to_list()
    holiday_rows = cal[cal.month.isin([10, 11, 12, 1])].index.to_list()
    delete_rows = early_rows + holiday_rows
    
    MIN_DAY = 'd_{}'.format(300)
    
    if 'd' in fr.columns: # d, series stack:
        fr = fr[fr.d >= day_to_cal_index[MIN_DAY]]
        fr = fr[~fr.d.isin([  day_to_cal_index[d] for d in delete_rows])]
        
        
    else:  # pivot table
        if MIN_DAY in fr.index:
            fr = fr.iloc[ fr.index.get_loc(MIN_DAY):, :]

        if len(delete_rows) > 0:
            fr = fr[~fr.index.isin(delete_rows)]
    
    return fr;


# In[115]:


def clean_features(features):
    for idx, feat_row in enumerate(features):
        fr = feat_row[1]
        fr = clean_df(fr)

        if len(fr) < len(feat_row[1]):
            features[idx] = (features[idx][0], fr)  


# In[ ]:





# In[116]:


ramCheck()


# In[117]:


print('Total Time Elapsed: ', (datetime.datetime.now() - start).seconds, 's')


# In[ ]:





# ### Cache Loader

# In[118]:


pickle_dir = '/kaggle/input/m5-e300/'

if CACHED_FEATURES:
    if 'features.pbz2' in os.listdir(pickle_dir):
        with bz2.BZ2File(pickle_dir + 'features.pbz2', 'r') as handle:
            features = pickle.load(handle)
    elif 'features.pgz' in os.listdir(pickle_dir):
        with gzip.GzipFile(pickle_dir + 'features.pgz', 'r') as handle:
            features = pickle.load(handle)
        


# In[119]:


ramCheck()


# In[120]:


print('Total Time Elapsed: ', (datetime.datetime.now() - start).seconds, 's')


# In[ ]:





# ### Clean Features

# In[121]:


clean_features(features)


# In[122]:


# clean_features(item_features)


# In[123]:


print('Total Time Elapsed: ', (datetime.datetime.now() - start).seconds, 's')


# In[124]:


ramCheck()


# In[ ]:





# ### Save Caches

# In[125]:


if CACHE_FEATURES:
    with gzip.GzipFile('features.pgz', 'w') as handle:
        pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.path.getsize('features.pgz') / 1e9


# In[126]:


print('Total Time Elapsed: ', (datetime.datetime.now() - start).seconds, 's')


# In[ ]:





# In[127]:


print('Total Time Elapsed: ', (datetime.datetime.now() - start).seconds, 's')


# In[ ]:





# ### Calendar Features

# In[128]:


cal_features = pd.DataFrame()

cal_features['dayofweek'] =  cal.date.dt.dayofweek.astype(np.int8)
cal_features['dayofmonth'] =  cal.date.dt.day.astype(np.int8)
cal_features['season'] =  cal.date.dt.month.astype(np.float64)


# In[ ]:





# ### State Calendar Features

# In[129]:


state_cal_features = []


# In[130]:


snap_cols = [c for c in cal.columns if 'snap' in c]

state_cal_features.append( ( 'snap_day' , 
                                cal[snap_cols].astype(np.int8) ) )
state_cal_features.append( ( 'snap_day_lag_1' , 
                                cal[snap_cols].shift(1).fillna(0).astype(np.int8) ) )
state_cal_features.append( ( 'snap_day_lag_2' , 
                                cal[snap_cols].shift(2).fillna(0).astype(np.int8) ) )


# In[131]:


state_cal_features.append( ( 'nth_snap_day',
            (cal[snap_cols].rolling(15, min_periods = 1).sum() * cal[snap_cols] ).astype(np.int8)  ) )


# In[132]:


for window in [2, 5, 10, 30, 60]:
    state_cal_features.append( ('snap_{}d_ewm'.format(window),
                                    cal[snap_cols].ewm(span = window, adjust=False).mean().astype(np.float64) ) )


# In[133]:


# strip columns to match state_id
def snapRename(x):
    return x.replace('snap_', '')

for f in range(0, len(state_cal_features)):
    state_cal_features[f] = (state_cal_features[f][0],
                                state_cal_features[f][1].rename(snapRename, axis = 'columns')) 


# In[134]:


# pd.merge( pd.Series(np.sum(train_flipped, axis = 1), name='total_sales'), cal, 
#          left_index=True, right_index=True).groupby('event_name_2').mean()\
#                 .sort_values('total_sales', ascending=False)


# In[ ]:





# In[ ]:





# ### Holidays

# In[135]:


for etype in [c for c in cal.event_type_1.dropna().unique()]:
    cal[etype.lower() + '_holiday'] = np.where(cal.event_type_1 == etype,
                                       cal.event_name_1,
                                               np.where(cal.event_type_2 == etype,
                                                    cal.event_name_2, 'None'))

for etype in [c for c in cal.event_type_1.dropna().unique()]:
    cal[etype.lower() + '_holiday'] = cal[etype.lower() + '_holiday'].astype('category')


# In[ ]:





# ### Price Features

# In[136]:


def getPricePivot():
    prices = pd.read_csv(path+ '/' + 'sell_prices.csv',
                    dtype = {'wm_yr_wk': np.int16, 'sell_price': np.float64})
    prices['id'] = prices.item_id + "_" + prices.store_id
    price_pivot =  prices.pivot(columns = 'id' , index='wm_yr_wk', values = 'sell_price')
    return price_pivot


price_pivot = getPricePivot()


# In[137]:


ramCheck()


# In[138]:


# memCheck()


# In[ ]:





# ### Assemble Series-Features Matrix

# #### Dicts

# In[139]:


series_to_series_id = dict([(col, idx) for idx, col in enumerate(train_flipped.columns)])
series_id_to_series = dict([(idx, col) for idx, col in enumerate(train_flipped.columns)])
series_id_level = dict([(idx, col) for idx, col in enumerate(levels)])
series_level = dict(zip(train_flipped.columns, levels))

series_to_item_id = dict([(x[1].id, x[1].item_id) for x in train_head[['id', 'item_id']].iterrows()])


# In[ ]:





# #### Features

# In[140]:


for feature in features:
    assert feature[1].shape == features[0][1].shape


# In[141]:


fstack = features[0][1].stack(dropna = False)
series_features = pd.DataFrame({'d': fstack.index.get_level_values(0)                                                 .map(day_to_cal_index).values.astype(np.int16),
                     'series': fstack.index.get_level_values(1) \
                                                .map(series_to_series_id).values.astype(np.int16)  })
del fstack


# In[142]:


for idx, feature in enumerate(features):
    if feature is not None:
        series_features[feature[0]] = feature[1].stack(dropna=False).values
        
del features 


# In[ ]:





# In[143]:


ramCheck()


# In[ ]:





# #### State Cal Features

# In[144]:


for feature in state_cal_features:
    assert feature[1].shape == state_cal_features[0][1].shape


# In[145]:


fstack = state_cal_features[0][1].stack(dropna = False)


# In[146]:


state_cal_series_features = pd.DataFrame({'d': fstack.index.get_level_values(0)                                                 .map(day_to_cal_index).values.astype(np.int16),
                     'state': fstack.index.get_level_values(1)  })
del fstack


# In[147]:


for idx, feature in enumerate(state_cal_features):
    if feature is not None:
        state_cal_series_features[feature[0]] = feature[1].stack(dropna=False).values
        


# In[ ]:





# #### Clean Up NA

# In[148]:


series_features.isnull().sum().sum()


# In[149]:


series_features.fillna(-10, inplace=True)


# In[ ]:





# #### Add Categoricals

# In[150]:


CATEGORICALS = ['dept_id', 'cat_id', 'store_id', 'state_id', ] # 'item_id'] # never item_id; wrecks higher layers;

        
for col in CATEGORICALS:
    series_features[col] = series_features.series.map(series_id_to_series).map(
                train_head.set_index('id')[col]) #.astype('category')


# In[ ]:





# In[151]:


ramCheck()


# In[152]:


# memCheck()


# In[ ]:





# In[153]:


print('Total Time Elapsed: ', (datetime.datetime.now() - start).seconds, 's')


# In[ ]:





# ### Metrics and Scaling

# In[154]:


def addSuffix(c):
    return c + '_validation'


# In[155]:


trailing_28d_sales = daily_sales.iloc[:,6:].transpose().rolling(28, min_periods = 1).sum().astype(np.float64)

fstack = train_flipped.stack(dropna = False)
weight_stack = pd.DataFrame({'d': fstack.index.get_level_values(0)                                                 .map(day_to_cal_index).values.astype(np.int16),
                     'series': fstack.index.get_level_values(1) \
                                                .map(series_to_series_id).values.astype(np.int16),
                    'days_since_first': (~train_flipped.isnull()).expanding().sum().stack(dropna = False).values\
                                             .astype(np.int16),
                    'trailing_vol': ( (train_flipped.diff().abs()).expanding().mean() ).astype(np.float64)\
                                                 .stack(dropna = False).values,
                    'weights': (trailing_28d_sales / 
                                    trailing_28d_sales.transpose().groupby(levels).sum().loc[levels].transpose().values)
                                     .astype(np.float64)\
                                             .stack(dropna = False).values,
                            })

del fstack


# In[156]:


del trailing_28d_sales; 


# In[157]:


weight_stack.dtypes


# In[ ]:





# In[158]:


new_items = weight_stack.days_since_first < 30
weight_stack[new_items].weights.sum() / weight_stack[weight_stack.days_since_first >= 0].weights.sum()
weight_stack.loc[new_items, 'weights'] = 0


# In[ ]:





# In[159]:


ramCheck()


# In[160]:


print('Total Time Elapsed: ', (datetime.datetime.now() - start).seconds, 's')


# In[ ]:





# ### Merge Weight and Y into Main Df

# In[161]:


weight_stack = clean_df(weight_stack)


# In[162]:


assert len(weight_stack) == len(series_features)
assert (weight_stack.d.values == series_features.d).all()
assert (weight_stack.series.values == series_features.series).all()


# In[163]:


series_features = pd.concat( (series_features, 
                weight_stack.reset_index(drop=True).iloc[:, -2:]), axis = 1,)


# In[164]:


weight_stack = weight_stack.iloc[:10, :]


# In[ ]:





# In[165]:


fstack = train_flipped.stack(dropna = False)
y_full = pd.DataFrame({'d': fstack.index.get_level_values(0)                                                 .map(day_to_cal_index).values.astype(np.int16),
                     'series': fstack.index.get_level_values(1) \
                                                .map(series_to_series_id).values.astype(np.int16),
                      'y': fstack.values})
del fstack


# In[ ]:





# In[166]:


ramCheck()


# In[167]:


# memCheck()


# In[ ]:





# ### Feature Merges to Build X/Y/etc.

# In[168]:


def addMAcrosses(X):
    EWMS = [c for c in X.columns if 'ewm' in c and 'qs_' in c and len(c) < 12]
    for idx1, col1 in enumerate(EWMS):
        for idx2, col2 in enumerate(EWMS):
            if not idx1 < idx2:
                continue;
            
            X['qs_{}_{}_ewm_diff'.format(col1.split('_')[1], col2.split('_')[1])] = X[col1] - X[col2]
            X['qs_{}_{}_ewm_ratio'.format(col1.split('_')[1], col2.split('_')[1])] = X[col1] / X[col2]
                
    return X
    


# In[ ]:





# In[169]:


def addCalFeatures(X):  # large block of code; easy;
    # day of week, month, season of year
    X['dayofweek'] = ( X.d + X.days_fwd).map(cal_index_to_day).map(cal_features.dayofweek)
    X['dayofmonth'] = ( X.d + X.days_fwd).map(cal_index_to_day).map(cal_features.dayofmonth)
 
    X['basedayofweek'] = X.d.map(cal_index_to_day).map(cal_features.dayofweek)
    X['dayofweekchg'] = (X.days_fwd % 7).astype(np.int8)

    X['basedayofmonth'] = X.d.map(cal_index_to_day).map(cal_features.dayofmonth)
    X['season'] =  ( ( X.d + X.days_fwd).map(cal_index_to_day).map(cal_features.season) + np.random.normal( 0, 1, len(X))).astype(np.float64)
                        # with a full month SD of noise to not overfit to specific days;

    # holidays
    holiday_cols = [c for c in cal.columns if '_holiday' in c]
    for col in holiday_cols:
        X['base_' + col] = X.d.map(cal_index_to_day).map(cal[col])
        X[col] = ( X.d + X.days_fwd).map(cal_index_to_day).map(cal[col])

    
    return X
#     'dayofweek'


# In[170]:


def convertToLinearFeatures(X):
    X = X.copy()
    for s in X.dayofweek.unique():
        X['dayofweek_{}'.format(s)] = (X.dayofweek == s).astype(np.int8)
    X.drop( columns = X.columns[X.dtypes == 'category'], inplace=True)
    X['daysfwd_sqrt'] = (X.days_fwd ** 0.5).astype(np.float64)
    
    return X


# In[171]:


def addStateCalFeatures(X):  
    if (X.state_id == 'All').mean() > 0:
        print('No State Ids')
        return X;
    
    def rename_scf(c, name = 'basedate'):
        return c if (c=='d' or c == 'state') else name + '_' + c
    
    X['future_d'] = ( X.d + X.days_fwd)
    X['state'] = X.state_id.astype('object')
    
    nX = X.merge(state_cal_series_features[['state', 'd', 'snap_day', 'nth_snap_day']]
                 .rename(rename_scf, axis = 'columns'),
                                         on = ['d', 'state'],  
             validate='m:1', how = 'inner', suffixes = (False, False)) 
    
    
    nX = nX.merge(state_cal_series_features[['state', 'd', 'snap_day', 'nth_snap_day']]
                 .rename(columns = {'d': 'future_d'}), 
                                         on = ['future_d', 'state'],  
             validate='m:1', how = 'inner', suffixes = (False, False)) 
    
    nX.drop(columns = ['state', 'future_d'], inplace=True)
    
    assert len(nX) == len(X)
    
    
    
    return nX


# In[172]:


def add_item_features(X):  
    return X


# In[173]:
VALIDATION = -1  # 2016 # pure holdout from train and prediction sets;


# In[174]:
def getXYG(X, scale_range = None, oos = False):
    start_time = datetime.datetime.now(); 

    # ensure it's in the train set, and days_forward is actually *forward*
    X.drop( X.index[ (X.days_fwd < 1) |
           (  ~oos  &  ( X.d + X.days_fwd > cal.index.get_loc(train_flipped.index[-1])  )    ) ], inplace=True)
    g = gc.collect()
    
    
    X = addMAcrosses(X)

    X = addCalFeatures(X)
    X = addStateCalFeatures(X)
    
    # noise to time-static features
    for col in [c for c in X.columns if 'store' in c and 'ratio' in c]:
        X[col] = X[col] + np.random.normal(0, 0.1, len(X))
        print('adding noise to {}'.format(col))
    

    # match with Y
    if 'y' not in X.columns:
        st = datetime.datetime.now(); 
        X['future_d'] = X.d + X.days_fwd
        if oos:  
            X = X.merge(y_full.rename(columns = {'d': 'future_d'}), on = ['future_d', 'series'], 
                             how = 'left')
            X.y = X.y.fillna(-1)
            
        else:  
            X = X.merge(y_full.rename(columns = {'d': 'future_d'}), on = ['future_d', 'series'],
                       )#    suffixes = (None, None), validate = 'm:1')
#     X['yo'] = X.y.copy()
    g = gc.collect()
    
    scaler_columns = [c for c in X.columns if c in weight_stack.columns[2:]]
    scalers = X[scaler_columns].copy()
    y = X.y
    
    groups = pd.Series(cal.iloc[(X.d + X.days_fwd)].year.values, X.index).astype(np.int16)
    
    
    # feature drops
    if REDUCED_FEATURES:
        feat_drops = [c for c in X.columns if c not in (sparse_features + ['d', 'series', 'days_fwd'])]
    
    elif len(FEATURE_DROPS) > 0:
        feat_drops = [c for c in X.columns if any(z in c for z in FEATURE_DROPS )]
        print('dropping {} features; anything containing {}'.format(len(feat_drops), FEATURE_DROPS))
        print('   -- {}'.format(feat_drops))
    else:
        feat_drops = []
        
    # final drops
    X.drop(columns = scaler_columns + (['future_d'] if 'future_d' in X.columns else []) + ['y'] + feat_drops , inplace=True)

    scalers['scaler'] = scalers.trailing_vol.copy()
    
    # randomize scaling
    if scale_range > 0:
        scalers.scaler = scalers.scaler * np.exp( scale_range * ( np.random.normal(0, 0.5, len(X))) )
#         scalers.scaler = scalers.scaler * np.exp( scale_range * ( np.random.rand(len(X)) - 0.5) )
    
    # now rescale y and  'scaled variable' in X by its vol
    for col in [c for c in X.columns if 'qs_' in c and 'ratio' not in c]:
        X[col] = np.where( X[col] == -10, X[col], (X[col] / scalers.scaler).astype(np.float64)) 
    y = y / scalers.scaler
    
    
    yn = (oos == False) & (y.isnull() | (groups==VALIDATION)) 

    
    print("\nXYG Pull Time: {}".format(str(datetime.datetime.now() - start_time).split('.', 2)[0] ))
    
    return (X[~yn], y[~yn], groups[~yn], scalers[~yn])


# In[175]:
def getSubsample(frac, level = 12, scale_range=0.1, n_repeats = 1, drops = True, post_process_X = None):
    start_time = datetime.datetime.now(); 

    wtg_mean = series_features.weights[(series_features.series.map(series_id_level) == level)].mean()
    ss = series_features.weights / wtg_mean * frac
    
    X = series_features[ (ss > np.random.rand(len(ss)) )
                              & (series_features.series.map(series_id_level) == level) ]
    ss =  X.weights / wtg_mean * frac
      
    print('{} series that seek oversampling'.format( (ss > 1). sum() ) )
    print( ss[ss>1].sort_values()[-5:])
    
    extras = []
    
    while ss.max() > 1:
        ss = ss - 1
        extras.append( X[ ss > np.random.rand(len(ss))] )
        
    if len(extras) > 0:
        print(' scaled EWMS of extras:')
        print( ( extras[-1].qs_30d_ewm / extras[-1].trailing_vol)[-5:] )

    if len(extras) > 0:
        X = pd.concat((X, *extras))
    else:
        X = X.copy()
    
    
    X['days_fwd'] = (np.random.randint(0, 28, size = len(X)) + 1).astype(np.int8)
    
    if n_repeats > 1:
         X = pd.concat([X] * n_repeats)

    g = gc.collect()
    
    X, y, groups, scalers = getXYG(X, scale_range)
    ramCheck()
    g = gc.collect()
    if drops:
        X.drop(columns = ['d', 'series'], inplace=True)
    
    if post_process_X is not None:
        X = post_process_X(X)
    
    print(X.shape)
    print("\nSubsample Time: {}\n".format(str(datetime.datetime.now() - start_time).split('.', 2)[0] ))

    return X, y, groups, scalers


# In[176]:


print('Total Time Elapsed: ', (datetime.datetime.now() - start).seconds, 's')


# In[ ]:





# ### Modeling

# In[177]:


from sklearn.model_selection import RandomizedSearchCV, GroupKFold, LeaveOneGroupOut
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import make_scorer
import lightgbm as lgb


# In[ ]:





# In[178]:


def quantile_loss(true, pred, quantile = 0.5):
    loss = np.where(true >= pred, 
                        quantile*(true-pred),
                        (1-quantile)*(pred - true) )
    return np.mean(loss)   
 


# In[179]:


def quantile_scorer(quantile = 0.5):
    return make_scorer(quantile_loss, False, quantile = quantile)


# In[180]:


lgb_quantile_params = {     # fairly well tuned, with high runtimes 
                'max_depth': [10, 20],
                'n_estimators': [   200, 300, 350, 400, ],   
                'min_split_gain': [0, 0, 0, 0, 1e-4, 1e-3, 1e-2, 0.1],
                'min_child_samples': [ 2, 4, 7, 10, 14, 20, 30, 40, 60, 80, 100, 130, 170, 200, 300, 500, 700, 1000 ],
                'min_child_weight': [0, 0, 0, 0, 1e-4, 1e-3, 1e-3, 1e-3, 5e-3, 2e-2, 0.1 ],
                'num_leaves': [ 20, 30, 30, 30, 50, 70, 90, ],
                'learning_rate': [  0.02, 0.03, 0.04, 0.04, 0.05, 0.05, 0.07, ],         
                'colsample_bytree': [0.3, 0.5, 0.7, 0.8, 0.9, 0.9, 0.9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
                'colsample_bynode':[0.1, 0.15, 0.2, 0.2, 0.2, 0.25, 0.3, 0.5, 0.65, 0.8, 0.9, 1],
                'reg_lambda': [0, 0, 0, 0, 1e-5, 1e-5, 1e-5, 1e-5, 3e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100   ],
                'reg_alpha': [0, 1e-5, 3e-5, 1e-4, 1e-4, 1e-3, 3e-3, 1e-2, 0.1, 1, 1, 10, 10, 100, 1000,],
                'subsample': [  0.9, 1],
                'subsample_freq': [1],
                'cat_smooth': [0.1, 0.2, 0.5, 1, 2, 5, 7, 10],
}


# In[181]:


if SPEED or SUPER_SPEED or REDUCED_FEATURES:
    lgb_quantile_params = {     # fairly well tuned, with high runtimes 
                'max_depth': [10, 20],
                'n_estimators': [ 150, 200, 200],  # 300, 350, 400, ],   
                'min_split_gain': [0, 0, 0, 0, 1e-4, 1e-3, 1e-2, 0.1],
                'min_child_samples': [ 2, 4, 7, 10, 14, 20, 30, 40, 60, 80, 100, 100, 100, 
                                                  130, 170, 200, 300, 500, 700, 1000 ],
                'min_child_weight': [0, 0, 0, 0, 1e-4, 1e-3, 1e-3, 1e-3, 5e-3, 2e-2, 0.1 ],
                'num_leaves': [ 20, 30, 50, 50 ], # 50, 70, 90, ],
                'learning_rate': [  0.04, 0.05, 0.07, 0.07, 0.07, 0.1, 0.1, 0.1 ],   # 0.02, 0.03,        
                'colsample_bytree': [0.3, 0.5, 0.7, 0.8, 0.9, 0.9, 0.9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
                'colsample_bynode':[0.1, 0.15, 0.2, 0.2, 0.2, 0.25, 0.3, 0.5, 0.65, 0.8, 0.9, 1],
                'reg_lambda': [0, 0, 0, 0, 1e-5, 1e-5, 1e-5, 1e-5, 3e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100   ],
                'reg_alpha': [0, 1e-5, 3e-5, 1e-4, 1e-4, 1e-3, 3e-3, 1e-2, 0.1, 1, 1, 10, 10, 100, 1000,],
                'subsample': [  0.9, 1],
                'subsample_freq': [1],
                'cat_smooth': [0.1, 0.2, 0.5, 1, 2, 5, 7, 10],
    }


# In[ ]:





# In[182]:


def trainLGBquantile(x, y, groups, cv = 0, n_jobs = -1, alpha = 0.5, **kwargs):
    clfargs = kwargs.copy(); clfargs.pop('n_iter', None)
    clf = lgb.LGBMRegressor(verbosity=-1, hist_pool_size = 1000,  objective = 'quantile', alpha = alpha,
                            importance_type = 'gain',
                            seed = datetime.datetime.now().microsecond if TIME_SEED else None,
                             **clfargs,
                      )
    print('\n\n Running Quantile Regression for \u03BC={}\n'.format(alpha))
    params = lgb_quantile_params
    
    return trainModel(x, y, groups, clf, params, quantile_scorer(alpha), n_jobs, **kwargs)


# In[ ]:





# In[ ]:





# In[ ]:





# In[183]:


def trainModel(x, y, groups, clf, params, cv = 0, n_jobs = None, 
                   verbose=0, splits=None, **kwargs):
    if n_jobs is None:
        n_jobs = -1
    folds = LeaveOneGroupOut()
    clf = RandomizedSearchCV(clf, params, cv=  folds, 
                             n_iter= ( kwargs['n_iter'] if len(kwargs) > 0 and 'n_iter' in kwargs else 4), 
                            verbose = 0, n_jobs = n_jobs, scoring = cv)
    f = clf.fit(x, y, groups)
    print(pd.DataFrame(clf.cv_results_['mean_test_score'])); print();  

    best = clf.best_estimator_;  print(best)
    print("\nBest In-Sample CV: {}\n".format(np.round(clf.best_score_,4)))

    return best


# In[ ]:





# In[184]:


def runQBags(n_bags = 3, model_type = trainLGBquantile, data = None, quantiles = [0.5], **kwargs):
    start_time = datetime.datetime.now(); 
    
    clf_set = []; loss_set = []
    for bag in range(0, n_bags):
        print('\n\n  Running Bag {} of {}\n\n'.format(bag+1, n_bags))
        if data is None:
            X, y, groups, scalers = getSubsample()
        else:
            X, y, groups, scalers = data

        group_list = [*dict.fromkeys(groups)]   
        group_list.sort()
        print("Groups: {}".format(group_list))

        clfs = []; preds = []; ys=[]; datestack = []; losses = pd.DataFrame(index=QUANTILES)
        if SINGLE_FOLD: group_list = group_list[-1:]
        for group in group_list:
            print('\n\n   Running Models with {} Out-of-Fold\n\n'.format(group))
            x_holdout = X[groups == group]
            y_holdout = y[groups == group]
            
            ramCheck()
            model = model_type 
            
            q_clfs = []; q_losses = []
            for quantile in quantiles:
                set_filter = (groups != group) & (np.random.rand(len(groups)) <
                                 quantile_wts[quantile] ** (0.35 if LEVEL >=11 else 0.25) )
                clf = model(X[set_filter], y[set_filter], groups[set_filter], 
                                alpha = quantile, **kwargs) 
                q_clfs.append(clf)

                predicted = clf.predict(x_holdout)

                q_losses.append((quantile, quantile_loss(y_holdout, predicted, quantile)))
                print(u"{} \u03BC={:.3f}: {:.4f}".format(group, quantile, q_losses[-1][1] ) )
                
                preds.append(predicted)
                ys.append(y_holdout)
            
            clfs.append(q_clfs)
            print("\nLevel {} OOS Losses for Bag {} in {}:".format(level, bag+1, group))
            print(np.round(pd.DataFrame(q_losses).set_index(0)[1], 4))
            losses[group] = np.round(pd.DataFrame(q_losses).set_index(0)[1], 4).values
            print("\nElapsed Time So Far This Bag: {}\n".format(str(datetime.datetime.now() - start_time).split('.', 2)[0] ))
            
        
        clf_set.append(clfs)
        print("\nLevel {} Year-by-Year OOS Losses for Bag {}:".format(level, bag, group))
        print(losses)
        
        loss_set.append(losses)
        print("\nModel Bag Time: {}\n".format(str(datetime.datetime.now() - start_time).split('.', 2)[0] ))
    return clf_set, loss_set


# In[ ]:





# In[ ]:





# In[185]:


level_os = dict([(idx, 1/val) for (idx,val) in level_multiplier.items()])


# In[186]:


# these are to use less processing time on edge quantiles 
QUANTILE_LEVELS = [0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995]
QUANTILE_WTS = [0.1, 0.2, 0.6, 0.8, 1, 0.9, 0.7, 0.2, 0.1,]
    
quantile_wts = dict(zip(QUANTILE_LEVELS, QUANTILE_WTS))


# In[ ]:





# In[187]:


print('Total Time Elapsed: ', (datetime.datetime.now() - start).seconds, 's')


# In[ ]:





# ### Actually Run Model

# In[188]:


if not IMPORT:
    clf_set = {}; loss_set = {}; LEVEL_QUANTILES = {};
    for level in sorted(levels.unique()):
        print("\n\n\nRunning Models for Level {}\n\n\n".format(level))
        
        SS_FRAC, SCALE_RANGE = P_DICT[level] # if level < 12 else ID_FILTER]; 
        SS_FRAC = SS_FRAC * SS_SS
        print('{}/{}'.format(SS_FRAC, SCALE_RANGE))
        
        # much higher iteration counts for low levels
        clf_set[level], loss_set[level] = runQBags(n_bags = int(BAGS * level_os[level] ** BAGS_PWR), 
                                                   model_type = trainLGBquantile, 
                                                   data = getSubsample(SS_FRAC * level_os[level] ** SS_PWR, 
                                                                       level, SCALE_RANGE),
                                                        n_iter =  int( 
                                                                 (2.2 if level <= 9 else 1.66) 
                                                                   * (16 - (level if level <=12 else 12) ) 
                                                                    * (1/4 if SUPER_SPEED else (1/2 if SPEED else 1))   
                                                                     ) ,
                      quantiles = QUANTILES,
                       n_jobs = N_JOBS) 
        
        LEVEL_QUANTILES[level] = QUANTILES


# In[ ]:





# In[189]:


print('Total Time Elapsed: ', (datetime.datetime.now() - start).seconds, 's')


# In[ ]:





# In[ ]:





# In[ ]:





# ### Import Classifiers

# In[190]:


if IMPORT:
    clf_sets = []  # ***
    path = '/kaggle/input/m5clfs/'
    
   # if LEVEL != 12: 
    files = [f for f in os.listdir(path) if '.pkl' in f]
    if LEVEL == 13 and MAX_LEVEL is None: files = [f for f in files if '13_' in f or 'hobbies' in f]
    if LEVEL == 14 and MAX_LEVEL is None: files = [f for f in files if '14_' in f or 'household' in f]
    if LEVEL == 15 and MAX_LEVEL is None: files = [f for f in files if '15_' in f or 'foods' in f]      
        
  #  else:
  #      files = [f for f in os.listdir(path) if '.pkl' in f and ID_FILTER.lower() in f]
        
    for file in files:
        clf_sets.append(pickle.load(open(path + file,'rb')))
 
    clf_df = []; pairs = []
    for clf_set in clf_sets:
        for level, level_clfs in clf_set.items():
            for clf_bag_idx, clf_bag in enumerate(level_clfs):
                for group_idx, clf_group in enumerate(clf_bag):
                    for quantile_idx, clf in enumerate(clf_group):
                        clf_df.append((level, clf.alpha, group_idx, clf))


    clf_df = pd.DataFrame(clf_df, columns = ['level', 'alpha', 'group', 'clf'])
    
    if LEVEL > 12 and MAX_LEVEL == None:
        clf_df.loc[clf_df.level==12, 'level'] = LEVEL


    # clf_df
    
    LEVEL_QUANTILES = {}; clf_set = {}
    for level in sorted(clf_df.level.unique()):

        level_df = clf_df[clf_df.level == level]

        level_list = []
        for group in sorted(level_df.group.unique()):
            group_df = level_df[level_df.group == group].sort_values('alpha')
            if level in LEVEL_QUANTILES:
                assert LEVEL_QUANTILES[level] == list(group_df.alpha)
            else:
                LEVEL_QUANTILES[level] = list(group_df.alpha)
            level_list.append(list(group_df.clf))
        if len(level_df.group.unique()) > 1:
            SINGLE_FOLD = False
        clf_set[level] = [level_list]
        print(level, ": ", LEVEL_QUANTILES[level]); 


# In[191]:


# LEVEL


# In[ ]:





# ### Display

# In[192]:


for level in sorted(clf_set.keys()):
    print("Level {}:".format(level))
    
    for idx, q in enumerate(LEVEL_QUANTILES[level]):
        print(u'\n\n      Regressors for \u03BC={}:\n'.format(q))
        for clf in [q_clfs[idx] for clfs in clf_set[level] for q_clfs in clfs]:
            print(clf)
    
    print(); print()


# In[ ]:





# In[193]:


print('Total Time Elapsed: ', (datetime.datetime.now() - start).seconds, 's')


# In[ ]:





# In[194]:


# save classifiers
clf_file = ('clf_set.pkl' if IMPORT 
                          else ('lvl_{}_clfs.pkl'.format(LEVEL) if MAX_LEVEL == None 
                                                            else 'lvls_lt_{}_clfs.pkl'.format(MAX_LEVEL)))
with open(clf_file, 'wb') as handle:
    pickle.dump(clf_set, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[ ]:





# In[195]:


ramCheck()


# In[ ]:





# ### Feature Importance

# In[196]:


def show_FI(model, featNames, featCount):
   # show_FI_plot(model.feature_importances_, featNames, featCount)
    fis = model.feature_importances_
    fig, ax = plt.subplots(figsize=(6, 5))
    indices = np.argsort(fis)[::-1][:featCount]
    g = sns.barplot(y=featNames[indices][:featCount],
                    x = fis[indices][:featCount] , orient='h' )
    g.set_xlabel("Relative importance")
    g.set_ylabel("Features")
    g.tick_params(labelsize=12)
    g.set_title( " feature importance")
    


# In[197]:


def avg_FI(all_clfs, featNames, featCount, title = "Feature Importances"):
    # 1. Sum
    clfs = []
    for clf_set in all_clfs:
        for clf in clf_set:
            clfs.append(clf);
    fi = np.zeros( (len(clfs), len(clfs[0].feature_importances_)) )
    for idx, clf in enumerate(clfs):
        fi[idx, :] = clf.feature_importances_
    avg_fi = np.mean(fi, axis = 0)

    # 2. Plot
    fis = avg_fi
    fig, ax = plt.subplots(figsize=(6, 5))
    indices = np.argsort(fis)[::-1]#[:featCount]
    #print(indices)
    g = sns.barplot(y=featNames[indices][:featCount],
                    x = fis[indices][:featCount] , orient='h' )
    g.set_xlabel("Relative importance")
    g.set_ylabel("Features")
    g.tick_params(labelsize=12)
    g.set_title(title + ' - {} classifiers'.format(len(clfs)))
    
    return pd.Series(fis[indices], featNames[indices])


# In[198]:


def linear_FI_plot(fi, featNames, featCount):
   # show_FI_plot(model.feature_importances_, featNames, featCount)
    fig, ax = plt.subplots(figsize=(6, 5))
    indices = np.argsort(np.absolute(fi))[::-1]#[:featCount]
    g = sns.barplot(y=featNames[indices][:featCount],
                    x = fi[indices][:featCount] , orient='h' )
    g.set_xlabel("Relative importance")
    g.set_ylabel("Features")
    g.tick_params(labelsize=12)
    g.set_title( " feature importance")
    return pd.Series(fi[indices], featNames[indices])


# In[ ]:





# In[199]:


for level in sorted(clf_set.keys()):
    X = getSubsample(0.0001, level, 0.1)[0]
    print("Level {}:".format(level))
    for idx, q in enumerate(LEVEL_QUANTILES[level]):
        f = avg_FI([[q_clfs[idx] for clfs in clf_set[level] for q_clfs in clfs]], X.columns, 25, 
                       title = "Level {} \u03BC={} Feature Importances".format(level, q))
    print(); print()


# In[ ]:





# In[200]:


ramCheck()


# In[201]:


print('Total Time Elapsed: ', (datetime.datetime.now() - start).seconds, 's')


# In[ ]:





# In[ ]:





# ### Predict

# In[202]:


def avg(arr, axis = 0):
    return np.median(arr, axis = axis)


# In[203]:


def predictSet(X, y, groups, scalers, clf_set):
    start_time = datetime.datetime.now(); 
    
    group_list = [*dict.fromkeys(groups)]   
    group_list.sort()
#     print(group_list)
    
    y_unscaled = y * scalers.scaler
    
    all_preds = []; ys=[]; gs = []; xs = []; scaler_stack = []
    if SINGLE_FOLD: group_list = group_list[-1:]
    for group_idx, group in enumerate(group_list):
        g = gc.collect()
        x_holdout = X[groups == group]
        y_holdout = y_unscaled[groups == group] 
        scalers_holdout = scalers[groups == group]
        groups_holdout = groups[groups == group]
        
        preds = np.zeros( (len(QUANTILES), len(y_holdout)), dtype=np.float64)
        for q_idx, quantile in enumerate(QUANTILES):            
            q_preds = np.zeros( ( len(clf_set), len(y_holdout) ) )
            for bag_idx, clf in enumerate(clf_set):
                x_clean = x_holdout.drop(columns = [c for c in x_holdout.columns if c=='d' or c=='series'])
                if group_idx >= len(clf_set[bag_idx]): # if out of sample year, blend all years
                    qs_preds = np.zeros( (group_idx, len(x_clean)) )
                    for gidx in range(group_idx):
                        qs_preds[gidx, :] = clf_set[bag_idx][gidx][q_idx].predict(x_clean)
                    q_preds[bag_idx, :] = np.mean(qs_preds, axis = 0)
                else:
                    q_preds[bag_idx, :] = clf_set[bag_idx][group_idx][q_idx].predict(x_clean)
                
            q_preds = avg(q_preds) * scalers_holdout.scaler

            preds[q_idx, :] = q_preds
            
#             print(u"{} \u03BC={:.3f}: {:.4f}".format(group, quantile, quantile_loss(y_holdout, q_preds, quantile) ) )
        
        all_preds.append(preds)
        xs.append(x_holdout)
        ys.append(y_holdout)
        gs.append(groups_holdout)
        scaler_stack.append(scalers_holdout)
        print()
    y_pred = np.hstack(all_preds)
    scaler_stack = pd.concat(scaler_stack)
    y_true = pd.concat(ys)
    groups = pd.concat(gs)
    X = pd.concat(xs)
    
    end_time = datetime.datetime.now(); 
    print("Bag Prediction Time: {}".format(str(end_time - start_time).split('.', 2)[0] ))
    return y_pred, y_true, groups, scaler_stack, X


# In[204]:


def predictOOS(X, scalers, clf_set, QUANTILES, validation = False):
    start_time = datetime.datetime.now(); 
    
    group_list = [1 + i for i in range(0, len(clf_set[0]))]   
    if validation:
        group_list = np.zeros(len(clf_set[0]))
        group_list[-1] = 1
    
    
    divisor = sum(group_list)
    print(np.round([g / divisor for g in group_list], 3)); print()
    
    x_holdout = X
    scalers_holdout = scalers 

    preds = np.zeros( (len(clf_set[0][0]), len(x_holdout)), dtype=np.float64)
    for q_idx in range( len(clf_set[0][0])): # loop over quantiles
        print(u'Predicting for \u03BC={}'.format( QUANTILES[q_idx]) )
        
        q_preds = np.zeros( ( len(clf_set), len(x_holdout) ), dtype = np.float64 )
        for bag_idx, clf in enumerate(clf_set):
            x_clean = x_holdout # .drop(columns = [c for c in x_holdout.columns if c=='d' or c=='series'])
            qs_preds = np.zeros( (len(group_list), len(x_clean)), dtype = np.float64 )
            if SINGLE_FOLD: group_list = group_list[-1:]
            for gidx in range(len(group_list)):
                if group_list[gidx] > 0: 
                    qs_preds[gidx, :] = clf_set[bag_idx][gidx][q_idx].predict(x_clean) * group_list[gidx] / divisor
            q_preds[bag_idx, :] = np.sum(qs_preds, axis = 0)

        q_preds = np.mean(q_preds, axis = 0) * scalers_holdout.scaler

        preds[q_idx, :] = q_preds
 
    end_time = datetime.datetime.now(); 
    print("Bag Prediction Time: {}".format(str(end_time - start_time).split('.', 2)[0] ))
    return preds


# In[205]:


def wspl(true, pred, weights, trailing_vol, quantile = 0.5):
    loss = weights * np.where(true >= pred, 
                        quantile*(true-pred),
                        (1-quantile)*(pred - true) ) / trailing_vol
    return np.mean(loss) / np.mean(weights)   
 


# In[ ]:





# ### Random Sample Scoring

# In[210]:


VALIDATION = -1


# In[211]:


RSEED = 11


# In[212]:


# number of samples for each data point;
N_REPEATS = 20 #if LE <15 else 10  


# In[213]:


# clf_set


# In[214]:


qls = {}; all_predictions = {}
for level in sorted(set(clf_set.keys()) & set(levels)):
    print("\n\n\nLevel {}\n\n\n".format(level))
    QUANTILES = LEVEL_QUANTILES[level]
    
    SS_FRAC, SCALE_RANGE = P_DICT[level] #  if level < 12 else ID_FILTER]; 
    SS_FRAC = SS_FRAC * SS_SS 
    EVAL_FRAC = SS_FRAC * (1 if level < 11 else 1/2) 
    EVAL_PWR = 0.6
    SCALE_RANGE_TEST = SCALE_RANGE
    
    np.random.seed(RSEED)
    X, y, groups, scalers = getSubsample(EVAL_FRAC * level_os[level] ** EVAL_PWR, level, 
                                         SCALE_RANGE_TEST, 
                                         n_repeats = N_REPEATS if level < 15 else N_REPEATS//2, 
                                         drops=False)
    if len(X) == 0:
        print("No Data for Level {}".format(level))
        continue;
        
    y_pred, y_true, groups, scaler_stack, X = predictSet(X, y, groups, scalers, clf_set[level]); 
   # assert (y_true == y.values * scalers.trailing_vol).all()

    predictions = pd.DataFrame(y_pred.T, index=y_true.index, columns = QUANTILES)
    predictions['y_true'] = y_true.values
    predictions = pd.concat((predictions, scaler_stack), axis = 'columns')
    predictions['group'] = groups.values
    predictions['series'] = X.series
    predictions['d'] = X.d
    predictions['days_fwd'] = X.days_fwd
    
    
    
    losses = pd.DataFrame(index=QUANTILES)
    for group in groups.unique():
        subpred = predictions[predictions.group == group]
        q_losses = []
        for quantile in QUANTILES:
            q_losses.append((quantile, wspl(subpred.y_true, subpred[quantile], 
                                  1, subpred.trailing_vol, quantile)))
        losses[group] = np.round(pd.DataFrame(q_losses).set_index(0)[1], 4).values
    qls[level] = [losses]    
    
    ramCheck()
    
    # now combine them
    predictions = predictions.groupby(['series', 'd', 'days_fwd']).agg(
                dict([(col, 'mean') for col in predictions.columns 
                          if col not in ['series', 'd', 'days_fwd']]\
                         + [('days_fwd', 'count')])  )\
            .rename(columns = {'days_fwd': 'ct'}).reset_index()
    predictions.head()
    predictions.sort_values('ct', ascending = False).head(5)
    print(len(predictions))
    
    all_predictions[level] = predictions


# In[215]:


for level in sorted(all_predictions.keys()):
    predictions = all_predictions[level]
    
    losses = pd.DataFrame(index=LEVEL_QUANTILES[level])
    for group in groups.unique():
        subpred = predictions[predictions.group == group]
        q_losses = []
        for quantile in QUANTILES:
            q_losses.append((quantile, wspl(subpred.y_true, subpred[quantile], 
                                  subpred.ct, subpred.trailing_vol, quantile)))
        losses[group] = np.round(pd.DataFrame(q_losses).set_index(0)[1], 4).values
        
        
    qls[level] = [losses]
    
    print("\n\n\nLevel {} Year-by-Year OOS Losses for Evaluation Bag {}:".format(level, 1))
    print(losses); #print(); print()
        
#     print(BAGS)
#     print(SS_FRAC)
#     print(X.shape); #del X
#     print(SCALE_RANGE_TEST)
#     print(N_REPEATS)
    
    


# In[251]:


# all_predictions[1][all_predictions[1].d == 1912].drop(columns = ['series', 'd', 'group', 'ct'])\
#     .set_index('days_fwd').plot()


# In[217]:


# X.dayofweek


# In[218]:


for level in sorted(all_predictions.keys()):
#     print("\nLevel {}:".format(level))
    predictions = all_predictions[level]

    predictions['future_d'] = predictions.d + predictions.days_fwd

    for quantile in QUANTILES:
        true = predictions.y_true
        pred = predictions[quantile]
        trailing_vol= predictions.trailing_vol

        predictions['loss_{}'.format(quantile)] =              np.where(true >= pred, 
                            quantile*(true-pred),
                            (1-quantile)*(pred - true) ) / trailing_vol

    predictions['loss'] = predictions[[c for c in predictions.columns if 'loss_' in str(c)]].sum(axis = 1)  
    predictions['wtg_loss'] = predictions.loss * predictions.ct / predictions.ct.mean()    

    # predictions.groupby('series').loss.sum()
    # predictions.groupby('series').wtg_loss.sum()
    # predictions.groupby('series').wtg_loss.sum().sum()

#     predictions.groupby(['series', 'd']).wtg_loss.sum().reset_index().pivot('d', 'series', values='wtg_loss').plot()

#     predictions.groupby(['series', 'd']).wtg_loss.sum().reset_index().pivot('d', 'series', values='wtg_loss')\
#             .ewm(span = 7).mean().plot();

#     (predictions.groupby(['series', 'future_d']).wtg_loss.sum().reset_index()\
#                 .pivot('future_d', 'series', values='wtg_loss').ewm(span = 7).mean() \
#     ).plot();

    # predictions.groupby(['series', 'future_d']).wtg_loss.sum().sort_values(ascending = False) #.ewm(span = 7).mean() \
    # ).plot();
    # predictions.groupby(['series', 'future_d']).wtg_loss.sum().sum()

#     predictions[(predictions.series == 0) & (predictions.days_fwd < 7 )].groupby('future_d').mean()\
#             [[c for c in predictions.columns if '.' in str(c) and 'loss' not in str(c)]]\
#                 .loc[1550:1700].plot(linewidth = 0.4)
#     train_flipped.iloc[:, 1].reset_index(drop=True).loc[1550:1700].plot( linewidth = 1);
    # train_flipped.iloc[active_days, 1].iloc[1000:].plot();


# In[ ]:





# In[ ]:





# In[ ]:





# In[219]:


print('Total Time Elapsed: ', (datetime.datetime.now() - start).seconds, 's')


# In[220]:


ramCheck()


# In[221]:


# memCheck()


# In[ ]:





# ### Make Submission

# In[222]:


MEM_CAPACITY = 3e6  


# In[223]:


MAX_RUNS = 2500 * (1/10 if SPEED or SUPER_SPEED else 1)
MIN_RUNS = 20 * (1/20 if SPEED or SUPER_SPEED else 1)


# In[224]:


all_predictions = {}
for level in sorted(list(set(levels.unique()) & set(clf_set.keys()))):
    print('\n\nCreating Out-of-Sample Predictions for Level {}\n'.format(level))
    
    final_base = FINAL_BASE

    assert (final_base in ['d_1941', 'd_1913'])
    if final_base == 'd_1941':
        suffix = 'evaluation'
    elif final_base == 'd_1913':
        suffix = 'validation'
        
    print('   predicting 28 days forward from {}'.format(final_base))
    final_features = series_features[( series_features.d.map(cal_index_to_day) == final_base) & 
                                         (series_features.series.map(series_id_level) == level) ]

    print('    for {} series'.format(len(final_features)))
    
    SS_FRAC, SCALE_RANGE = P_DICT[level] # if level < 12 else ID_FILTER]; 
    SS_FRAC = SS_FRAC * 0.8
    print('   scale range of {}'.format(SCALE_RANGE))
    
    
    if level <= 9 or SPEED:
        X = []
        for df in range(0,28):
            Xi = final_features.copy()
            Xi['days_fwd'] = df + 1
            X.append(Xi)
        X = pd.concat(X, ignore_index = True); del Xi; del final_features;

        Xn = np.power(X.weights, 2)
        Xn = (Xn * MEM_CAPACITY / Xn.sum()).clip(MIN_RUNS, MAX_RUNS)
        Xn = (Xn * MEM_CAPACITY / Xn.sum()).clip(MIN_RUNS, MAX_RUNS)
        
        print('   average repeats: {:.0f}'.format(Xn.mean()))
        print('   median repeats: {:.0f}'.format(Xn.median()))
        print('   max repeats: {:.0f}'.format(Xn.max()))

        X = X.loc[np.repeat(Xn.index, Xn)]

        X, y, groups, scalers = getXYG(X, scale_range = SCALE_RANGE, oos = True)
        Xd = X.d;  Xseries = X.series
        X.drop(columns=['d', 'series'], inplace = True)

        print(X.shape)
        y_pred = predictOOS(X, scalers, clf_set[level], LEVEL_QUANTILES[level], suffix == 'validation'); print()

        predictions = pd.DataFrame(y_pred.T, index=X.index, columns = LEVEL_QUANTILES[level])
        predictions = pd.concat((predictions, scalers), axis = 'columns')
        predictions['series'] = Xseries
        predictions['d'] = Xd
        predictions['days_fwd'] = X.days_fwd.astype(np.int8)
        predictions['y_true'] = y * scalers.scaler
#         break;
        ramCheck()

        predictions = predictions.groupby(['series', 'd', 'days_fwd']).agg(
                        dict([(col, 'mean') for col in predictions.columns 
                                  if col not in ['series', 'd', 'days_fwd']]\
                                 + [('days_fwd', 'count')])  )\
                    .rename(columns = {'days_fwd': 'ct'}).reset_index()
        predictions.days_fwd = predictions.days_fwd.astype(np.int8)

    else: # levels 10, 11, 12
        
        predictions_full = []
        
        for df in range(0,28):
            print( '\n Predicting {} days forward from {}'.format(df + 1, final_base))
            X = final_features.copy()
            X['days_fwd'] = df + 1

            Xn = np.power(X.weights, 1.5)
            Xn = (Xn * MEM_CAPACITY / Xn.sum()).clip(MIN_RUNS, MAX_RUNS)
            Xn = (Xn * MEM_CAPACITY / Xn.sum()).clip(MIN_RUNS, MAX_RUNS)
            
            print('   average repeats: {:.0f}'.format(Xn.mean()))
            print('   median repeats: {:.0f}'.format(Xn.median()))
            print('   max repeats: {:.0f}'.format(Xn.max()))
            
            print('------------------')
            print(Xn.isna().values.any())
            Xn.fillna((Xn.mean(skipna=True)), inplace=True)
            print(Xn.isna().values.any())
            print('-----------------')

            X = X.loc[np.repeat(Xn.index, Xn)]

            X, y, groups, scalers = getXYG(X, scale_range = SCALE_RANGE, oos = True)
            Xd = X.d;  Xseries = X.series
            X.drop(columns=['d', 'series'], inplace = True)

            print(X.shape)
            y_pred = predictOOS(X, scalers, clf_set[level], LEVEL_QUANTILES[level], suffix == 'validation'); print()

            predictions = pd.DataFrame(y_pred.T, index=X.index, columns = LEVEL_QUANTILES[level])
            predictions = pd.concat((predictions, scalers), axis = 'columns')
            predictions['series'] = Xseries
            predictions['d'] = Xd
            predictions['days_fwd'] = X.days_fwd.astype(np.int8)
            predictions['y_true'] = y * scalers.scaler

            ramCheck()

            predictions = predictions.groupby(['series', 'd', 'days_fwd']).agg(
                            dict([(col, 'mean') for col in predictions.columns 
                                      if col not in ['series', 'd', 'days_fwd']]\
                                     + [('days_fwd', 'count')])  )\
                        .rename(columns = {'days_fwd': 'ct'}).reset_index()
            predictions.days_fwd = predictions.days_fwd.astype(np.int8)
            predictions_full.append(predictions)
            
        predictions = pd.concat(predictions_full); del predictions_full
 
    all_predictions[level] = predictions; del predictions


# In[225]:


with open('all_predictions_raw.pkl', 'wb') as handle:
    pickle.dump(all_predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[ ]:





# In[226]:


# all_predictions = pickle.load(open('../input/m5-submissions/all_predictions_valid_19.pkl', 'rb'))


# In[227]:



losses = pd.DataFrame(index=LEVEL_QUANTILES[levels.min()])
for level in sorted(all_predictions.keys()):
    predictions = all_predictions[level]
    subpred = predictions
    q_losses = []
    for quantile in LEVEL_QUANTILES[level]:
        q_losses.append((quantile, wspl(subpred.y_true, subpred[quantile], 
                              subpred.weights, subpred.trailing_vol, quantile)))

#         print(np.round(pd.DataFrame(q_losses).set_index(0)[1], 4).values)
    losses[level] = np.round(pd.DataFrame(q_losses).set_index(0)[1], 4).values


#         print("\n\n\nLevel {} Year-by-Year OOS Losses for Evaluation Bag {}:".format(level, 1))
print(losses); print(); print()
print(losses.mean())
print(losses.mean().mean())


# In[ ]:





# ### Level Harmonizer

# In[228]:


a = pd.DataFrame(index = range(1, 29))
for level in sorted(all_predictions.keys()):
    if level > 9:
        continue;
    a[level] = all_predictions[level].groupby('days_fwd')[0.5].sum() / level_multiplier[level]


# In[229]:


try:
    a.plot()
except:
    pass;


# In[ ]:





# In[230]:


# all_predictions[level][quantile]

# all_predictions[level][quantile] * all_predictions[level].days_fwd.map(a.mean(axis=1) / a[level] )


# In[231]:


ADJUSTMENT_FACTOR = 1 if SPEED or SUPER_SPEED else 0.7  # probably better as 1.0, but used 0.7 to be safe;


# In[232]:


for level in sorted(all_predictions.keys()):
    if level > 9: 
        continue;
        
    for quantile in LEVEL_QUANTILES[level]:
        all_predictions[level][quantile] = all_predictions[level][quantile]                         * ( (1 - ADJUSTMENT_FACTOR) +
                              ADJUSTMENT_FACTOR * all_predictions[level].days_fwd.map(  a.mean(axis=1) / a[level] ) )


# In[233]:


a = pd.DataFrame(index = range(1, 29))
for level in sorted(all_predictions.keys()):
    if level > 9:
        continue;
    a[level] = all_predictions[level].groupby('days_fwd')[0.5].sum() / level_multiplier[level]


# In[234]:


try:
    a.plot()
except:
    pass;


# In[ ]:





# In[235]:



losses = pd.DataFrame(index=LEVEL_QUANTILES[level])
for level in sorted(all_predictions.keys()):
    predictions = all_predictions[level]
    subpred = predictions
    q_losses = []
    for quantile in LEVEL_QUANTILES[level]:
        q_losses.append((quantile, wspl(subpred.y_true, subpred[quantile], 
                              subpred.weights, subpred.trailing_vol, quantile)))

#         print(np.round(pd.DataFrame(q_losses).set_index(0)[1], 4).values)
    losses[level] = np.round(pd.DataFrame(q_losses).set_index(0)[1], 4).values


#         print("\n\n\nLevel {} Year-by-Year OOS Losses for Evaluation Bag {}:".format(level, 1))
print(losses); print(); print()
print(losses.mean())
print(losses.mean().mean())


# In[ ]:





# In[236]:


if suffix == 'validation':

    losses = pd.DataFrame(index=LEVEL_QUANTILES[level])
    for level in sorted(all_predictions.keys()):
        predictions = all_predictions[level]
        subpred = predictions
        q_losses = []
        for quantile in LEVEL_QUANTILES[level]:
            q_losses.append((quantile, wspl(subpred.y_true, subpred[quantile], 
                                  subpred.weights, subpred.trailing_vol, quantile)))
        
        losses[level] = np.round(pd.DataFrame(q_losses).set_index(0)[1], 4).values


#         print("\n\n\nLevel {} Year-by-Year OOS Losses for Evaluation Bag {}:".format(level, 1))
    print(losses); print(); print()
    print(losses.mean())


# In[237]:


if suffix == 'validation':
    losses.plot()


# In[ ]:





# In[ ]:





# In[238]:


for level in sorted(all_predictions.keys()):
    predictions = all_predictions[level]
    (predictions.groupby('days_fwd')[0.5].sum() / level_multiplier[level]).plot(legend = True, 
                                                                                label = level,
                                                                               linewidth = 0.5)
    
if suffix=='validation':
    ( predictions.groupby('days_fwd').y_true.sum() / level_multiplier[level]) .plot(linewidth = 1.5)


# In[239]:


train_flipped.shape


# In[ ]:





# ### Graphs

# In[240]:


# (series_features[( series_features.d.map(cal_index_to_day) == final_base) & 
#                                          (series_features.series.map(series_id_level) == level) ]\
#         .sort_values('weights', ascending = False).reset_index().weights.astype(np.float64) ** 1.5).cumsum().plot()


# In[241]:


for level in sorted(all_predictions.keys()):
    predictions = all_predictions[level]
    
    if level <= 9:
        series_list = predictions.series.unique()[:5]
    else:
        series_list =  series_features[( series_features.d.map(cal_index_to_day) == final_base) & 
                                         (series_features.series.map(series_id_level) == level) ]\
            .sort_values('weights', ascending = False).series.to_list()\
                 [:len(predictions.series.unique())//20 : len(predictions.series.unique()) // 500]
    
    for series in series_list:
        
        DAYS_BACK = 60
        if suffix == 'evaluation':
            prior = train_flipped.iloc[-DAYS_BACK:, series]
            prior.index = range(-DAYS_BACK + 1, 1 )
        else:
            prior = train_flipped.iloc[-DAYS_BACK:, series]
            prior.index = range(-DAYS_BACK + 28 + 1, 28 + 1 )
            
            
        f = prior.plot( linewidth = 1.5);

        f = predictions[predictions.series == series].set_index('days_fwd')                [[c for c in predictions.columns if c in LEVEL_QUANTILES[level]]].plot(
                                title = ("Level {} - {}".format(level, series_id_to_series[series])
                                      + ("" if level <=9 else " - weight of {:.2%}".format(
                                          predictions[predictions.series == series].weights.mean() )))
                                                       , 
                                              linewidth = 0.5, ax = f);
        f = plt.figure();
#     break;


# In[ ]:





# In[ ]:





# In[242]:


output_rows = []
for level in sorted(all_predictions.keys()):
    predictions = all_predictions[level]
    df = predictions[ ['series', 'days_fwd'] + list(LEVEL_QUANTILES[level])].copy()
    df.series = df.series.map(series_id_to_series)
    df = df.melt(['series', 'days_fwd'], var_name = 'q' )
    df.value = df.value / level_multiplier[level]
    df['name'] = df.series + '_' + df.q.apply(lambda x: '{0:.3f}'.format(x)) + '_' + suffix
    # df.days_fwd = 'F' + df.days_fwd.astype(str)

    for q in df.q.unique():
        qdf = df[df.q==q].pivot('name', 'days_fwd', 'value')
        qdf.columns = ['F{}'.format(c) for c in qdf.columns]
        qdf.index.name = 'id'
        output_rows.append(qdf)
    output = pd.concat(output_rows)


# In[ ]:





# In[243]:


output.tail()


# In[ ]:





# In[244]:


sample_sub.head()


# In[245]:


assert len(set(output.index) - set(sample_sub.id)) == 0

assert len(set(sample_sub.id) & set(output.index)) == len(output)


# In[ ]:





# In[246]:


output_file = ('submission_{}_lvl_{}.csv'.format(suffix, LEVEL) if MAX_LEVEL == None 
                                else 'submission_{}_lt_{}.csv'.format(suffix, MAX_LEVEL))


# In[247]:


output.round(3).to_csv(output_file)


# In[248]:


print(len(output) )


# In[ ]:





# In[ ]:





# In[249]:


output


# In[250]:


print('Total Time Elapsed: ', (datetime.datetime.now() - start).seconds, 's')


# In[ ]:





# In[ ]:





# In[ ]:




