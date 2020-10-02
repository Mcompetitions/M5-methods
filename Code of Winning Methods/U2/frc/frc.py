# General imports
import numpy as np
import pandas as pd
import os, sys, gc, time, warnings, pickle, psutil, random

from math import ceil

from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

import sklearn
import lightgbm
import sys
print('py',sys.version)
{'numpy':np.__version__, 'pandas':pd.__version__,  'sklearn':sklearn.__version__, 'lightgbm':lightgbm.__version__, }

## Simple "Memory profilers" to see memory usage
def get_memory_usage():
    return np.round(psutil.Process(os.getpid()).memory_info()[0]/2.**30, 2) 
        
def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

## Memory Reducer
# :df pandas dataframe to reduce size             # type: pd.DataFrame()
# :verbose                                        # type: bool
def reduce_mem_usage(df, verbose=True):
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

## Merging by concat to not lose dtypes
def merge_by_concat(df1, df2, merge_on):
    merged_gf = df1[merge_on]
    merged_gf = merged_gf.merge(df2, on=merge_on, how='left')
    new_columns = [col for col in list(merged_gf) if col not in merge_on]
    df1 = pd.concat([df1, merged_gf[new_columns]], axis=1)
    return df1

########################### Vars
#################################################################################
TARGET = 'sales'         # Our main target
END_TRAIN = 1941         # Last day in train set
MAIN_INDEX = ['id','d']  # We can identify item by these columns

########################### Load Data
#################################################################################
print('Load Main Data')

# Here are reafing all our data 
# without any limitations and dtype modification
train_df = pd.read_csv('../raw/sales_train_evaluation.csv')
prices_df = pd.read_csv('../raw/sell_prices.csv')
calendar_df = pd.read_csv('../raw/calendar.csv')

train_df = train_df.loc[:,:f'd_{END_TRAIN}']

########################### Make Grid
#################################################################################
print('Create Grid')

# We can tranform horizontal representation 
# to vertical "view"
# Our "index" will be 'id','item_id','dept_id','cat_id','store_id','state_id'
# and labels are 'd_' coulmns

index_columns = ['id','item_id','dept_id','cat_id','store_id','state_id']
grid_df = pd.melt(train_df, 
                  id_vars = index_columns, 
                  var_name = 'd', 
                  value_name = TARGET)

# If we look on train_df we se that 
# we don't have a lot of traning rows
# but each day can provide more train data
print('Train rows:', len(train_df), len(grid_df))

# To be able to make predictions
# we need to add "test set" to our grid
add_grid = pd.DataFrame()
for i in range(1,29):
    temp_df = train_df[index_columns]
    temp_df = temp_df.drop_duplicates()
    temp_df['d'] = 'd_'+ str(END_TRAIN+i)
    temp_df[TARGET] = np.nan
    add_grid = pd.concat([add_grid,temp_df])

grid_df = pd.concat([grid_df,add_grid])
grid_df = grid_df.reset_index(drop=True)

# Remove some temoprary DFs
del temp_df, add_grid

# We will not need original train_df
# anymore and can remove it
del train_df

# You don't have to use df = df construction
# you can use inplace=True instead.
# like this
# grid_df.reset_index(drop=True, inplace=True)

# Let's check our memory usage
print("{:>20}: {:>8}".format('Original grid_df',sizeof_fmt(grid_df.memory_usage(index=True).sum())))

# We can free some memory 
# by converting "strings" to categorical
# it will not affect merging and 
# we will not lose any valuable data
for col in index_columns:
    grid_df[col] = grid_df[col].astype('category')

# Let's check again memory usage
print("{:>20}: {:>8}".format('Reduced grid_df',sizeof_fmt(grid_df.memory_usage(index=True).sum())))

########################### Product Release date
#################################################################################
print('Release week')

# It seems that leadings zero values
# in each train_df item row
# are not real 0 sales but mean
# absence for the item in the store
# we can safe some memory by removing
# such zeros

# Prices are set by week
# so it we will have not very accurate release week 
release_df = prices_df.groupby(['store_id','item_id'])['wm_yr_wk'].agg(['min']).reset_index()
release_df.columns = ['store_id','item_id','release']

# Now we can merge release_df
grid_df = merge_by_concat(grid_df, release_df, ['store_id','item_id'])
del release_df

# We want to remove some "zeros" rows
# from grid_df 
# to do it we need wm_yr_wk column
# let's merge partly calendar_df to have it
grid_df = merge_by_concat(grid_df, calendar_df[['wm_yr_wk','d']], ['d'])
                      
# Now we can cutoff some rows 
# and safe memory 
grid_df = grid_df[grid_df['wm_yr_wk']>=grid_df['release']]
grid_df = grid_df.reset_index(drop=True)

# Let's check our memory usage
print("{:>20}: {:>8}".format('Original grid_df',sizeof_fmt(grid_df.memory_usage(index=True).sum())))

# Should we keep release week 
# as one of the features?
# Only good CV can give the answer.
# Let's minify the release values.
# Min transformation will not help here 
# as int16 -> Integer (-32768 to 32767)
# and our grid_df['release'].max() serves for int16
# but we have have an idea how to transform 
# other columns in case we will need it
grid_df['release'] = grid_df['release'] - grid_df['release'].min()
grid_df['release'] = grid_df['release'].astype(np.int16)

# Let's check again memory usage
print("{:>20}: {:>8}".format('Reduced grid_df',sizeof_fmt(grid_df.memory_usage(index=True).sum())))

########################### Save part 1
#################################################################################
print('Save Part 1')

# We have our BASE grid ready
# and can save it as pickle file
# for future use (model training)
grid_df.to_pickle(f'grid_part_1_d{END_TRAIN}.pkl')

print('Size:', grid_df.shape)

########################### Prices
#################################################################################
print('Prices')

# We can do some basic aggregations
prices_df['price_max'] = prices_df.groupby(['store_id','item_id'])['sell_price'].transform('max')
prices_df['price_min'] = prices_df.groupby(['store_id','item_id'])['sell_price'].transform('min')
prices_df['price_std'] = prices_df.groupby(['store_id','item_id'])['sell_price'].transform('std')
prices_df['price_mean'] = prices_df.groupby(['store_id','item_id'])['sell_price'].transform('mean')

# and do price normalization (min/max scaling)
prices_df['price_norm'] = prices_df['sell_price']/prices_df['price_max']

# Some items are can be inflation dependent
# and some items are very "stable"
prices_df['price_nunique'] = prices_df.groupby(['store_id','item_id'])['sell_price'].transform('nunique')
prices_df['item_nunique'] = prices_df.groupby(['store_id','sell_price'])['item_id'].transform('nunique')

# I would like some "rolling" aggregations
# but would like months and years as "window"
calendar_prices = calendar_df[['wm_yr_wk','month','year']]
calendar_prices = calendar_prices.drop_duplicates(subset=['wm_yr_wk'])
prices_df = prices_df.merge(calendar_prices[['wm_yr_wk','month','year']], on=['wm_yr_wk'], how='left')
del calendar_prices

# Now we can add price "momentum" (some sort of)
# Shifted by week 
# by month mean
# by year mean
prices_df['price_momentum'] = prices_df['sell_price']/prices_df.groupby(['store_id','item_id'])['sell_price'].transform(lambda x: x.shift(1))
prices_df['price_momentum_m'] = prices_df['sell_price']/prices_df.groupby(['store_id','item_id','month'])['sell_price'].transform('mean')
prices_df['price_momentum_y'] = prices_df['sell_price']/prices_df.groupby(['store_id','item_id','year'])['sell_price'].transform('mean')

del prices_df['month'], prices_df['year']

########################### Merge prices and save part 2
#################################################################################
print('Merge prices and save part 2')

# Merge Prices
original_columns = list(grid_df)
grid_df = grid_df.merge(prices_df, on=['store_id','item_id','wm_yr_wk'], how='left')
keep_columns = [col for col in list(grid_df) if col not in original_columns]
grid_df = grid_df[MAIN_INDEX+keep_columns]
grid_df = reduce_mem_usage(grid_df)

# Safe part 2
grid_df.to_pickle(f'grid_part_2_d{END_TRAIN}.pkl')
print('Size:', grid_df.shape)

# We don't need prices_df anymore
del prices_df

# We can remove new columns
# or just load part_1
grid_df = pd.read_pickle(f'grid_part_1_d{END_TRAIN}.pkl')

########################### Merge calendar
#################################################################################
grid_df = grid_df[MAIN_INDEX]

# Merge calendar partly
icols = ['date',
         'd',
         'event_name_1',
         'event_type_1',
         'event_name_2',
         'event_type_2',
         'snap_CA',
         'snap_TX',
         'snap_WI']

grid_df = grid_df.merge(calendar_df[icols], on=['d'], how='left')

# Minify data
# 'snap_' columns we can convert to bool or int8
icols = ['event_name_1',
         'event_type_1',
         'event_name_2',
         'event_type_2',
         'snap_CA',
         'snap_TX',
         'snap_WI']
for col in icols:
    grid_df[col] = grid_df[col].astype('category')

# Convert to DateTime
grid_df['date'] = pd.to_datetime(grid_df['date'])

# Make some features from date
grid_df['tm_d'] = grid_df['date'].dt.day.astype(np.int8)
grid_df['tm_w'] = grid_df['date'].dt.week.astype(np.int8)
grid_df['tm_m'] = grid_df['date'].dt.month.astype(np.int8)
grid_df['tm_y'] = grid_df['date'].dt.year
grid_df['tm_y'] = (grid_df['tm_y'] - grid_df['tm_y'].min()).astype(np.int8)
grid_df['tm_wm'] = grid_df['tm_d'].apply(lambda x: ceil(x/7)).astype(np.int8)

grid_df['tm_dw'] = grid_df['date'].dt.dayofweek.astype(np.int8)
grid_df['tm_w_end'] = (grid_df['tm_dw']>=5).astype(np.int8)

# Remove date
del grid_df['date']

########################### Save part 3 (Dates)
#################################################################################
print('Save part 3')

# Safe part 3
grid_df.to_pickle(f'grid_part_3_d{END_TRAIN}.pkl')
print('Size:', grid_df.shape)

# We don't need calendar_df anymore
del calendar_df
del grid_df

########################### Some additional cleaning
#################################################################################

## Part 1
# Convert 'd' to int
grid_df = pd.read_pickle(f'grid_part_1_d{END_TRAIN}.pkl')
grid_df['d'] = grid_df['d'].apply(lambda x: x[2:]).astype(np.int16)

# Remove 'wm_yr_wk'
# as test values are not in train set
del grid_df['wm_yr_wk']
grid_df.to_pickle(f'grid_part_1_d{END_TRAIN}.pkl')

del grid_df

########################### Summary
#################################################################################

# Now we have 3 sets of features
grid_df = pd.concat([pd.read_pickle(f'grid_part_1_d{END_TRAIN}.pkl'),
                     pd.read_pickle(f'grid_part_2_d{END_TRAIN}.pkl').iloc[:,2:],
                     pd.read_pickle(f'grid_part_3_d{END_TRAIN}.pkl').iloc[:,2:]],
                     axis=1)
                     
# Let's check again memory usage
print("{:>20}: {:>8}".format('Full Grid',sizeof_fmt(grid_df.memory_usage(index=True).sum())))
print('Size:', grid_df.shape)

# 2.5GiB + is is still too big to train our model
# (on kaggle with its memory limits)
# and we don't have lag features yet
# But what if we can train by state_id or shop_id?
state_id = 'CA'
grid_df = grid_df[grid_df['state_id']==state_id]
print("{:>20}: {:>8}".format('Full Grid',sizeof_fmt(grid_df.memory_usage(index=True).sum())))
#           Full Grid:   1.2GiB

store_id = 'CA_1'
grid_df = grid_df[grid_df['store_id']==store_id]
print("{:>20}: {:>8}".format('Full Grid',sizeof_fmt(grid_df.memory_usage(index=True).sum())))
#           Full Grid: 321.2MiB

# Seems its good enough now
# In other kernel we will talk about LAGS features
# Thank you.

########################### Final list of features
#################################################################################
grid_df.info()

# General imports
import numpy as np
import pandas as pd
import os, sys, gc, time, warnings, pickle, psutil, random

import time

warnings.filterwarnings('ignore')

## Simple "Memory profilers" to see memory usage
def get_memory_usage():
    return np.round(psutil.Process(os.getpid()).memory_info()[0]/2.**30, 2) 
        
def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

########################### Vars
#################################################################################
TARGET = 'sales'         # Our main target
END_TRAIN = 1941       # Last day in train set
MAIN_INDEX = ['id','d']  # We can identify item by these columns

########################### Load Data
#################################################################################
print('Load Main Data')

# We will need only train dataset
# to show lags concept
train_df = pd.read_csv('../raw/sales_train_evaluation.csv')

# To make all calculations faster
# we will limit dataset by 'CA' state
train_df = train_df[train_df['state_id']=='CA']

########################### Data Representation
#################################################################################

# Let's check our shape
print('Shape', train_df.shape)

## Horizontal representation

# If we feed directly this data to model
# our label will be values in column 'd_1913'
# all other columns will be our "features"

# In lag terminology all d_1->d_1912 columns
# are our lag features 
# (target values in previous time period)

# Good thing that we have a lot of features here
# Bad thing is that we have just 12196 "train rows"
# Note: here and after all numbers are limited to 'CA' state
train_df.iloc[:10]

## Vertical representation

# In other hand we can think of d_ columns
# as additional labels and can significantly 
# scale up our training set to 23330948 rows

# Good thing that our model will have 
# greater input for training
# Bad thing that we are losing lags that we had
# in horizontal representation and
# also new data set consumes much more memory

index_columns = ['id','item_id','dept_id','cat_id','store_id','state_id']
train_df = pd.melt(train_df, 
                  id_vars = index_columns, 
                  var_name = 'd', 
                  value_name = TARGET)

train_df[train_df['id']=='HOBBIES_1_001_CA_1_validation'].iloc[:10]

## Some minification
train_df['d'] = train_df['d'].apply(lambda x: x[2:]).astype(np.int16)

icols = ['id','item_id','dept_id','cat_id','store_id','state_id']
for col in icols:
    train_df[col] = train_df[col].astype('category')

########################### Lags creation
#################################################################################

# We have several "code" solutions here
# As our dataset is allready sorted by d values
# we can simply shift() values
# also we have to keep in mind that 
# we need to aggregate values on 'id' level

# group and shift in loop
temp_df = train_df[['id','d',TARGET]]

start_time = time.time()
for i in range(1,8):
    print('Shifting:', i)
    temp_df['lag_'+str(i)] = temp_df.groupby(['id'])[TARGET].transform(lambda x: x.shift(i))
    
print('%0.2f min: Time for loops' % ((time.time() - start_time) / 60))


# Or same in "compact" manner
LAG_DAYS = [col for col in range(1,8)]
temp_df = train_df[['id','d',TARGET]]

start_time = time.time()
temp_df = temp_df.assign(**{
        '{}_lag_{}'.format(col, l): temp_df.groupby(['id'])[col].transform(lambda x: x.shift(l))
        for l in LAG_DAYS
        for col in [TARGET]
    })

print('%0.2f min: Time for bulk shift' % ((time.time() - start_time) / 60))

# The result
temp_df[temp_df['id']=='HOBBIES_1_001_CA_1_validation'].iloc[:10]

# You can notice many NaNs values - it's normal
# because there is no data for day 0,-1,-2
# (out of dataset time periods)

# Same works for test set
# be careful to make lag features:
# for day 1920 there is no data about day 1919 (until 1913)
# So if you want to predict day 1915 your 
# lag features have to start from 2 
# (1915(predicting day) - 1913(last day with label in dataset))
# and so on.

# There are few options to work 
# with NaNs in train set
## 1. drop it train_df[train_df['d']>MAX_LAG_DAY] 
## 1.1 in our case we already dropped some lines by release date
##     so you have find d.min() for each id
##     and drop train_df[train_df['d']>(train_df['d_min']+MAX_LAG_DAY)] 
## 2. If you want to keep it you can 
##    fill with '-1' to be able to convert to int
## 3. Leave as it is
## 4. Fill with mean -> not recommended

########################### Rolling lags
#################################################################################

# We restored some day sales values from horizontal representation
# as lag features but just few of them (last 7 days or less)
# because of memory limits we can't have many lag features
# How we can get additional information from other days?

## Rolling aggragations

temp_df = train_df[['id','d','sales']]

start_time = time.time()

for i in [14,30,60]:
    print('Rolling period:', i)
    temp_df['rolling_mean_'+str(i)] = temp_df.groupby(['id'])[TARGET].transform(lambda x: x.shift(1).rolling(i).mean())
    temp_df['rolling_std_'+str(i)]  = temp_df.groupby(['id'])[TARGET].transform(lambda x: x.shift(1).rolling(i).std())

# lambda x: x.shift(1)
# 1 day shift will serve only to predict day 1914
# for other days you have to shift PREDICT_DAY-1913

# Such aggregations will help us to restore
# at least part of the information for our model
# and out of 14+30+60->104 columns we can have just 6
# with valuable information (hope it is sufficient)
# you can also aggregate by max/skew/median etc 
# also you can try other rolling periods 180,365 etc
print('%0.2f min: Time for loop' % ((time.time() - start_time) / 60))


# The result
temp_df[temp_df['id']=='HOBBIES_1_002_CA_1_validation'].iloc[:20]

# Same for NaNs values - it's normal
# because there is no data for 
# 0*(rolling_period),-1*(rolling_period),-2*(rolling_period)

########################### Memory ussage
#################################################################################
# Let's check our memory usage
print("{:>20}: {:>8}".format('Original rolling df',sizeof_fmt(temp_df.memory_usage(index=True).sum())))

# can we minify it?
# 1. if our dataset are aligned by index 
#    you don't need 'id' 'd' 'sales' columns
temp_df = temp_df.iloc[:,3:]
print("{:>20}: {:>8}".format('Values rolling df',sizeof_fmt(temp_df.memory_usage(index=True).sum())))

# can we make it even smaller?
# carefully change dtype and/or
# use sparce matrix to minify 0s
# Also note that lgbm accepts matrixes as input
# that is good for memory reducion 
from scipy import sparse 
temp_matrix = sparse.csr_matrix(temp_df)

# restore to df
temp_matrix_restored = pd.DataFrame(temp_matrix.todense())
restored_cols = ['roll_' + str(i) for i in list(temp_matrix_restored)]
temp_matrix_restored.columns = restored_cols

########################### Remove old objects
#################################################################################
del temp_df, train_df, temp_matrix, temp_matrix_restored

########################### Apply on grid_df
#################################################################################
# lets read grid from 
# https://www.kaggle.com/kyakovlev/m5-simple-fe
# to be sure that our grids are aligned by index
grid_df = pd.read_pickle(f'./grid_part_1_d{END_TRAIN}.pkl')

# We need only 'id','d','sales'
# to make lags and rollings
grid_df = grid_df[['id','d','sales']]
SHIFT_DAY = 28

# Lags
# with 28 day shift
start_time = time.time()
print('Create lags')

LAG_DAYS = [col for col in range(SHIFT_DAY,SHIFT_DAY+15)]
grid_df = grid_df.assign(**{
        '{}_lag_{}'.format(col, l): grid_df.groupby(['id'])[col].transform(lambda x: x.shift(l))
        for l in LAG_DAYS
        for col in [TARGET]
    })

# Minify lag columns
for col in list(grid_df):
    if 'lag' in col:
        grid_df[col] = grid_df[col].astype(np.float16)

print('%0.2f min: Lags' % ((time.time() - start_time) / 60))

# Rollings
# with 28 day shift
start_time = time.time()
print('Create rolling aggs')

for i in [7,14,30,60,180]:
    print('Rolling period:', i)
    grid_df['rolling_mean_'+str(i)] = grid_df.groupby(['id'])[TARGET].transform(lambda x: x.shift(SHIFT_DAY).rolling(i).mean()).astype(np.float16)
    grid_df['rolling_std_'+str(i)]  = grid_df.groupby(['id'])[TARGET].transform(lambda x: x.shift(SHIFT_DAY).rolling(i).std()).astype(np.float16)

# Rollings
# with sliding shift
for d_shift in [1,7,14]: 
    print('Shifting period:', d_shift)
    for d_window in [7,14,30,60]:
        col_name = 'rolling_mean_tmp_'+str(d_shift)+'_'+str(d_window)
        grid_df[col_name] = grid_df.groupby(['id'])[TARGET].transform(lambda x: x.shift(d_shift).rolling(d_window).mean()).astype(np.float16)
    
    
print('%0.2f min: Lags' % ((time.time() - start_time) / 60))

########################### Export
#################################################################################
print('Save lags and rollings')
grid_df.to_pickle('lags_df_'+str(SHIFT_DAY)+f'_d{END_TRAIN}.pkl')

########################### Final list of new features
#################################################################################
grid_df.info()



## In this kernel I would like to show: 
## 1. FE creation approaches
## 2. Sequential fe validation
## 3. Dimension reduction
## 4. FE validation by Permutation importance
## 5. Mean encodings
## 6. Parallelization for FE

import numpy as np 
import pandas as pd 
import os, sys, gc, warnings, psutil, random

warnings.filterwarnings('ignore')

########################### Load data
########################### Basic features were created here:
########################### https://www.kaggle.com/kyakovlev/m5-simple-fe
#################################################################################

END_TRAIN = 1941  
# Read data
grid_df = pd.concat([pd.read_pickle(f'./grid_part_1_d{END_TRAIN}.pkl'),
                     pd.read_pickle(f'./grid_part_2_d{END_TRAIN}.pkl').iloc[:,2:],
                     pd.read_pickle(f'./grid_part_3_d{END_TRAIN}.pkl').iloc[:,2:]],
                     axis=1)

# Subsampling
# to make all calculations faster.
# Keep only 5% of original ids.
keep_id = np.array_split(list(grid_df['id'].unique()), 20)[0]
grid_df = grid_df[grid_df['id'].isin(keep_id)].reset_index(drop=True)

# Let's "inspect" our grid DataFrame
grid_df.info()

########################### Baseline model
#################################################################################

# We will need some global VARS for future

SEED = 42             # Our random seed for everything
random.seed(SEED)     # to make all tests "deterministic"
np.random.seed(SEED)
N_CORES = psutil.cpu_count()     # Available CPU cores

TARGET = 'sales'      # Our Target
    # And we will use last 28 days as validation

# Drop some items from "TEST" set part (1914...)
grid_df = grid_df[grid_df['d']<=END_TRAIN].reset_index(drop=True)

# Features that we want to exclude from training
remove_features = ['id','d',TARGET]

# Our baseline model serves
# to do fast checks of
# new features performance 

# We will use LightGBM for our tests
import lightgbm as lgb
lgb_params = {
                    'boosting_type': 'gbdt',         # Standart boosting type
                    'objective': 'regression',       # Standart loss for RMSE
                    'metric': ['rmse'],              # as we will use rmse as metric "proxy"
                    'subsample': 0.8,                
                    'subsample_freq': 1,
                    'learning_rate': 0.05,           # 0.5 is "fast enough" for us
                    'num_leaves': 2**7-1,            # We will need model only for fast check
                    'min_data_in_leaf': 2**8-1,      # So we want it to train faster even with drop in generalization 
                    'feature_fraction': 0.8,
                    'n_estimators': 5000,            # We don't want to limit training (you can change 5000 to any big enough number)
                    'early_stopping_rounds': 30,     # We will stop training almost immediately (if it stops improving) 
                    'seed': SEED,
                    'verbose': -1,
                } 

## RMSE
def rmse(y, y_pred):
    return np.sqrt(np.mean(np.square(y - y_pred)))

# Small function to make fast features tests
# estimator = make_fast_test(grid_df)
# it will return lgb booster for future analisys
def make_fast_test(df):

    features_columns = [col for col in list(df) if col not in remove_features]

    tr_x, tr_y = df[df['d']<=(END_TRAIN-28)][features_columns], df[df['d']<=(END_TRAIN-28)][TARGET]              
    vl_x, v_y = df[df['d']>(END_TRAIN-28)][features_columns], df[df['d']>(END_TRAIN-28)][TARGET]
    
    train_data = lgb.Dataset(tr_x, label=tr_y)
    valid_data = lgb.Dataset(vl_x, label=v_y)
    
    estimator = lgb.train(
                            lgb_params,
                            train_data,
                            valid_sets = [train_data,valid_data],
                            verbose_eval = 500,
                        )
    
    return estimator

# Make baseline model
baseline_model = make_fast_test(grid_df)

########################### Lets test our normal Lags (7 days)
########################### Some more info about lags here:
########################### https://www.kaggle.com/kyakovlev/m5-lags-features
#################################################################################

# Small helper to make lags creation faster
from multiprocessing import Pool                # Multiprocess Runs

## Multiprocessing Run.
# :t_split - int of lags days                   # type: int
# :func - Function to apply on each split       # type: python function
# This function is NOT 'bulletproof', be carefull and pass only correct types of variables.
## Multiprocess Runs
def df_parallelize_run(func, t_split):
    num_cores = np.min([N_CORES,len(t_split)])
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, t_split), axis=1)
    pool.close()
    pool.join()
    return df

def make_normal_lag(lag_day):
    lag_df = grid_df[['id','d',TARGET]] # not good to use df from "global space"
    col_name = 'sales_lag_'+str(lag_day)
    lag_df[col_name] = lag_df.groupby(['id'])[TARGET].transform(lambda x: x.shift(lag_day)).astype(np.float16)
    return lag_df[[col_name]]

# Launch parallel lag creation
# and "append" to our grid
LAGS_SPLIT = [col for col in range(1,1+7)]
#grid_df = pd.concat([grid_df, df_parallelize_run(make_normal_lag,LAGS_SPLIT)], axis=1)
grid_df = pd.concat([grid_df] + [make_normal_lag(x) for x in LAGS_SPLIT], axis=1)

# Make features test
test_model = make_fast_test(grid_df)

########################### Permutation importance Test
########################### https://www.kaggle.com/dansbecker/permutation-importance @dansbecker
#################################################################################

# Let's creat validation dataset and features
features_columns = [col for col in list(grid_df) if col not in remove_features]
validation_df = grid_df[grid_df['d']>(END_TRAIN-28)].reset_index(drop=True)

# Make normal prediction with our model and save score
validation_df['preds'] = test_model.predict(validation_df[features_columns])
base_score = rmse(validation_df[TARGET], validation_df['preds'])
print('Standart RMSE', base_score)


# Now we are looping over all our numerical features
for col in features_columns:
    
    # We will make validation set copy to restore
    # features states on each run
    temp_df = validation_df.copy()
    
    # Error here appears if we have "categorical" features and can't 
    # do np.random.permutation without disrupt categories
    # so we need to check if feature is numerical
    if temp_df[col].dtypes.name != 'category':
        temp_df[col] = np.random.permutation(temp_df[col].values)
        temp_df['preds'] = test_model.predict(temp_df[features_columns])
        cur_score = rmse(temp_df[TARGET], temp_df['preds'])
        
        # If our current rmse score is less than base score
        # it means that feature most probably is a bad one
        # and our model is learning on noise
        print(col, np.round(cur_score - base_score, 4))

# Remove Temp data
del temp_df, validation_df

# Remove test features
# As we will compare performance with baseline model for now
keep_cols = [col for col in list(grid_df) if 'sales_lag_' not in col]
grid_df = grid_df[keep_cols]


# Results:
## Lags with 1 days shift (nearest past) are important
## Some other features are not important and probably just noise
## Better make several Permutation runs to confirm useless of the feature
## link again https://www.kaggle.com/dansbecker/permutation-importance @dansbecker

## price_nunique -0.002 : strong negative values are most probably noise
## price_max -0.0002 : values close to 0 need deeper investigation


# from eli5 documentation (seems it's perfect explanation)

# The idea is the following: feature importance can be measured by looking at how much the score (accuracy, mse, rmse, mae, etc. - any score we’re interested in) decreases when a feature is not available.

# To do that one can remove feature from the dataset, re-train the estimator and check the score. But it requires re-training an estimator for each feature, which can be computationally intensive. Also, it shows what may be important within a dataset, not what is important within a concrete trained model.

# To avoid re-training the estimator we can remove a feature only from the test part of the dataset, and compute score without using this feature. It doesn’t work as-is, because estimators expect feature to be present. So instead of removing a feature we can **replace it with random noise** - feature column is still there, but it no longer contains useful information. This method works if noise is drawn from the **same distribution as original feature values** (as otherwise estimator may fail). The simplest way to get such noise is to shuffle values for a feature, i.e. use other examples’ feature values - this is how permutation importance is computed.

# ---

# It's not good when feature remove (replaced by noise) but we have better score. Simple and easy.

########################### Lets test far away Lags (7 days with 56 days shift)
########################### and check permutation importance
#################################################################################

LAGS_SPLIT = [col for col in range(56,56+7)]
grid_df = pd.concat([grid_df] + [make_normal_lag(x) for x in LAGS_SPLIT], axis=1)
test_model = make_fast_test(grid_df)

features_columns = [col for col in list(grid_df) if col not in remove_features]
validation_df = grid_df[grid_df['d']>(END_TRAIN-28)].reset_index(drop=True)
validation_df['preds'] = test_model.predict(validation_df[features_columns])
base_score = rmse(validation_df[TARGET], validation_df['preds'])
print('Standart RMSE', base_score)

for col in features_columns:
    temp_df = validation_df.copy()
    if temp_df[col].dtypes.name != 'category':
        temp_df[col] = np.random.permutation(temp_df[col].values)
        temp_df['preds'] = test_model.predict(temp_df[features_columns])
        cur_score = rmse(temp_df[TARGET], temp_df['preds'])
        print(col, np.round(cur_score - base_score, 4))

del temp_df, validation_df
        
# Remove test features
# As we will compare performance with baseline model for now
keep_cols = [col for col in list(grid_df) if 'sales_lag_' not in col]
grid_df = grid_df[keep_cols]


# Results:
## Lags with 56 days shift (far away past) are not as important
## as nearest past lags
## and at some point will be just noise for our model

########################### PCA
#################################################################################

# The main question here - can we have 
# almost same rmse boost with less features
# less dimensionality?

# Lets try PCA and make 7->3 dimensionality reduction

# PCA is "unsupervised" learning
# and with shifted target we can be sure
# that we have no Target leakage
from sklearn.decomposition import PCA

def make_pca(df, pca_col, n_days):
    print('PCA:', pca_col, n_days)
    
    # We don't need any other columns to make pca
    pca_df = df[[pca_col,'d',TARGET]]
    
    # If we are doing pca for other series "levels" 
    # we need to agg first
    if pca_col != 'id':
        merge_base = pca_df[[pca_col,'d']]
        pca_df = pca_df.groupby([pca_col,'d'])[TARGET].agg(['sum']).reset_index()
        pca_df[TARGET] = pca_df['sum']
        del pca_df['sum']
    
    # Min/Max scaling
    pca_df[TARGET] = pca_df[TARGET]/pca_df[TARGET].max()
    
    # Making "lag" in old way (not parallel)
    LAG_DAYS = [col for col in range(1,n_days+1)]
    format_s = '{}_pca_'+pca_col+str(n_days)+'_{}'
    pca_df = pca_df.assign(**{
            format_s.format(col, l): pca_df.groupby([pca_col])[col].transform(lambda x: x.shift(l))
            for l in LAG_DAYS
            for col in [TARGET]
        })
    
    pca_columns = list(pca_df)[3:]
    pca_df[pca_columns] = pca_df[pca_columns].fillna(0)
    pca = PCA(random_state=SEED)
    
    # You can use fit_transform here
    pca.fit(pca_df[pca_columns])
    pca_df[pca_columns] = pca.transform(pca_df[pca_columns])
    
    print(pca.explained_variance_ratio_)
    
    # we will keep only 3 most "valuable" columns/dimensions 
    keep_cols = pca_columns[:3]
    print('Columns to keep:', keep_cols)
    
    # If we are doing pca for other series "levels"
    # we need merge back our results to merge_base df
    # and only than return resulted df
    # I'll skip that step here
    
    return pca_df[keep_cols]


# Make PCA
grid_df = pd.concat([grid_df, make_pca(grid_df,'id',7)], axis=1)

# Make features test
test_model = make_fast_test(grid_df)

# Remove test features
# As we will compare performance with baseline model for now
keep_cols = [col for col in list(grid_df) if '_pca_' not in col]
grid_df = grid_df[keep_cols]

########################### Mean/std target encoding
#################################################################################

# We will use these three columns for test
# (in combination with store_id)
icols = ['item_id','cat_id','dept_id']

# But we can use any other column or even multiple groups
# like these ones
#            'state_id',
#            'store_id',
#            'cat_id',
#            'dept_id',
#            ['state_id', 'cat_id'],
#            ['state_id', 'dept_id'],
#            ['store_id', 'cat_id'],
#            ['store_id', 'dept_id'],
#            'item_id',
#            ['item_id', 'state_id'],
#            ['item_id', 'store_id']

# There are several ways to do "mean" encoding
## K-fold scheme
## LOO (leave one out)
## Smoothed/regularized 
## Expanding mean
## etc 

# You can test as many options as you want
# and decide what to use
# Because of memory issues you can't 
# use many features.

# We will use simple target encoding
# by std and mean agg
for col in icols:
    print('Encoding', col)
    temp_df = grid_df[grid_df['d']<=(1913-28)] # to be sure we don't have leakage in our validation set
    
    temp_df = temp_df.groupby([col,'store_id']).agg({TARGET: ['std','mean']})
    joiner = '_'+col+'_encoding_'
    temp_df.columns = [joiner.join(col).strip() for col in temp_df.columns.values]
    temp_df = temp_df.reset_index()
    grid_df = grid_df.merge(temp_df, on=[col,'store_id'], how='left')
    del temp_df

# Make features test
test_model = make_fast_test(grid_df)

# Remove test features
keep_cols = [col for col in list(grid_df) if '_encoding_' not in col]
grid_df = grid_df[keep_cols]

# Bad thing that for some items  
# we are using past and future values.
# But we are looking for "categorical" similiarity
# on a "long run". So future here is not a big problem.

########################### Last non O sale
#################################################################################

def find_last_sale(df,n_day):
    
    # Limit initial df
    ls_df = df[['id','d',TARGET]]
    
    # Convert target to binary
    ls_df['non_zero'] = (ls_df[TARGET]>0).astype(np.int8)
    
    # Make lags to prevent any leakage
    ls_df['non_zero_lag'] = ls_df.groupby(['id'])['non_zero'].transform(lambda x: x.shift(n_day).rolling(2000,1).sum()).fillna(-1)

    temp_df = ls_df[['id','d','non_zero_lag']].drop_duplicates(subset=['id','non_zero_lag'])
    temp_df.columns = ['id','d_min','non_zero_lag']

    ls_df = ls_df.merge(temp_df, on=['id','non_zero_lag'], how='left')
    ls_df['last_sale'] = ls_df['d'] - ls_df['d_min']

    return ls_df[['last_sale']]


# Find last non zero
# Need some "dances" to fit in memory limit with groupers
grid_df = pd.concat([grid_df, find_last_sale(grid_df,1)], axis=1)

# Make features test
test_model = make_fast_test(grid_df)

# Remove test features
keep_cols = [col for col in list(grid_df) if 'last_sale' not in col]
grid_df = grid_df[keep_cols]

########################### Apply on grid_df
#################################################################################
# lets read grid from 
# https://www.kaggle.com/kyakovlev/m5-simple-fe
# to be sure that our grids are aligned by index
grid_df = pd.read_pickle(f'./grid_part_1_d{END_TRAIN}.pkl')
grid_df[TARGET][grid_df['d']>(1913-28)] = np.nan
base_cols = list(grid_df)

icols =  [
            ['state_id'],
            ['store_id'],
            ['cat_id'],
            ['dept_id'],
            ['state_id', 'cat_id'],
            ['state_id', 'dept_id'],
            ['store_id', 'cat_id'],
            ['store_id', 'dept_id'],
            ['item_id'],
            ['item_id', 'state_id'],
            ['item_id', 'store_id']
            ]

for col in icols:
    print('Encoding', col)
    col_name = '_'+'_'.join(col)+'_'
    grid_df['enc'+col_name+'mean'] = grid_df.groupby(col)[TARGET].transform('mean').astype(np.float16)
    grid_df['enc'+col_name+'std'] = grid_df.groupby(col)[TARGET].transform('std').astype(np.float16)

keep_cols = [col for col in list(grid_df) if col not in base_cols]
grid_df = grid_df[['id','d']+keep_cols]

#################################################################################
print('Save Mean/Std encoding')
grid_df.to_pickle(f'mean_encoding_df_d{END_TRAIN}.pkl')

########################### Final list of new features
#################################################################################
grid_df.info()



# General imports
import numpy as np
import pandas as pd
import os, sys, gc, time, warnings, pickle, psutil, random

# custom imports
from multiprocessing import Pool        # Multiprocess Runs


warnings.filterwarnings('ignore')

########################### Helpers
#################################################################################
## Seeder
# :seed to make all processes deterministic     # type: int
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
## Memory Reducer
# :df pandas dataframe to reduce size             # type: pd.DataFrame()
# :verbose                                        # type: bool
def reduce_mem_usage(df, verbose=True):
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

########################### Helper to load data by store ID
#################################################################################
# Read data
def get_data_by_store(store):
    
    # Read and contact basic feature
    df = pd.concat([pd.read_pickle(BASE),
                    pd.read_pickle(PRICE).iloc[:,2:],
                    pd.read_pickle(CALENDAR).iloc[:,2:]],
                    axis=1)
    
    # Leave only relevant store
    df = reduce_mem_usage(df[df['store_id']==store])

    # With memory limits we have to read 
    # lags and mean encoding features
    # separately and drop items that we don't need.
    # As our Features Grids are aligned 
    # we can use index to keep only necessary rows
    # Alignment is good for us as concat uses less memory than merge.
    df2 = reduce_mem_usage(pd.read_pickle(MEAN_ENC)[mean_features])
    df2 = df2[df2.index.isin(df.index)]
    
    df3 = reduce_mem_usage(pd.read_pickle(LAGS).iloc[:,3:])
    df3 = df3[df3.index.isin(df.index)]
    
    df = pd.concat([df, df2], axis=1)
    del df2 # to not reach memory limit 
    
    df = pd.concat([df, df3], axis=1)
    del df3 # to not reach memory limit 
    
    # Create features list
    features = [col for col in list(df) if col not in remove_features]
    df = df[['id','d',TARGET]+features]
    
    # Skipping first n rows
    df = reduce_mem_usage(df[df['d']>=START_TRAIN].reset_index(drop=True))
    
    return df, features

# Recombine Test set after training
def get_base_test():
    base_test = pd.DataFrame()

    for store_id in STORES_IDS:
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        temp_df = pd.read_pickle('test_'+store_id+f'_d{END_TRAIN}_v2.pkl')
        #temp_df = pd.read_pickle('test_'+store_id+'.pkl')
        temp_df['store_id'] = store_id
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
                    'learning_rate': 0.03,
                    'num_leaves': 2**11-1,
                    'min_data_in_leaf': 2**12-1,
                    'feature_fraction': 0.5,
                    'max_bin': 100,
                    'n_estimators': 1400,
                    'boost_from_average': False,
                    'verbose': -1,
                } 

# Let's look closer on params

## 'boosting_type': 'gbdt'
# we have 'goss' option for faster training
# but it normally leads to underfit.
# Also there is good 'dart' mode
# but it takes forever to train
# and model performance depends 
# a lot on random factor 
# https://www.kaggle.com/c/home-credit-default-risk/discussion/60921

## 'objective': 'tweedie'
# Tweedie Gradient Boosting for Extremely
# Unbalanced Zero-inflated Data
# https://arxiv.org/pdf/1811.10192.pdf
# and many more articles about tweediie
#
# Strange (for me) but Tweedie is close in results
# to my own ugly loss.
# My advice here - make OWN LOSS function
# https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/140564
# https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/143070
# I think many of you already using it (after poisson kernel appeared) 
# (kagglers are very good with "params" testing and tuning).
# Try to figure out why Tweedie works.
# probably it will show you new features options
# or data transformation (Target transformation?).

## 'tweedie_variance_power': 1.1
# default = 1.5
# set this closer to 2 to shift towards a Gamma distribution
# set this closer to 1 to shift towards a Poisson distribution
# my CV shows 1.1 is optimal 
# but you can make your own choice

## 'metric': 'rmse'
# Doesn't mean anything to us
# as competition metric is different
# and we don't use early stoppings here.
# So rmse serves just for general 
# model performance overview.
# Also we use "fake" validation set
# (as it makes part of the training set)
# so even general rmse score doesn't mean anything))
# https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/133834

## 'subsample': 0.5
# Serves to fight with overfit
# this will randomly select part of data without resampling
# Chosen by CV (my CV can be wrong!)
# Next kernel will be about CV

##'subsample_freq': 1
# frequency for bagging
# default value - seems ok

## 'learning_rate': 0.03
# Chosen by CV
# Smaller - longer training
# but there is an option to stop 
# in "local minimum"
# Bigger - faster training
# but there is a chance to
# not find "global minimum" minimum

## 'num_leaves': 2**11-1
## 'min_data_in_leaf': 2**12-1
# Force model to use more features
# We need it to reduce "recursive"
# error impact.
# Also it leads to overfit
# that's why we use small 
# 'max_bin': 100

## l1, l2 regularizations
# https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c
# Good tiny explanation
# l2 can work with bigger num_leaves
# but my CV doesn't show boost
                    
## 'n_estimators': 1400
# CV shows that there should be
# different values for each state/store.
# Current value was chosen 
# for general purpose.
# As we don't use any early stopings
# careful to not overfit Public LB.

##'feature_fraction': 0.5
# LightGBM will randomly select 
# part of features on each iteration (tree).
# We have maaaany features
# and many of them are "duplicates"
# and many just "noise"
# good values here - 0.5-0.7 (by CV)

## 'boost_from_average': False
# There is some "problem"
# to code boost_from_average for 
# custom loss
# 'True' makes training faster
# BUT carefull use it
# https://github.com/microsoft/LightGBM/issues/1514
# not our case but good to know cons



########################### Vars
#################################################################################
VER = 2                          # Our model version
SEED = 42                        # We want all things
seed_everything(SEED)            # to be as deterministic 
lgb_params['seed'] = SEED        # as possible
N_CORES = psutil.cpu_count()     # Available CPU cores


#LIMITS and const
TARGET      = 'sales'            # Our target
START_TRAIN = 0                  # We can skip some rows (Nans/faster training)
END_TRAIN   = 1941              # End day of our train set
P_HORIZON   = 28                 # Prediction horizon
USE_AUX     = False              # Use or not pretrained models

#FEATURES to remove
## These features lead to overfit
## or values not present in test set
remove_features = ['id','state_id','store_id',
                   'date','wm_yr_wk','d',TARGET]
mean_features   = ['enc_cat_id_mean','enc_cat_id_std',
                   'enc_dept_id_mean','enc_dept_id_std',
                   'enc_item_id_mean','enc_item_id_std'] 

#PATHS for Features
ORIGINAL = '../raw/'
BASE     = f'./grid_part_1_d{END_TRAIN}.pkl'
PRICE    = f'./grid_part_2_d{END_TRAIN}.pkl'
CALENDAR = f'./grid_part_3_d{END_TRAIN}.pkl'
LAGS     = f'./lags_df_28_d{END_TRAIN}.pkl'
MEAN_ENC = f'./mean_encoding_df_d{END_TRAIN}.pkl'


# AUX(pretrained) Models paths
AUX_MODELS = './'


#STORES ids
STORES_IDS = pd.read_csv(ORIGINAL+'sales_train_evaluation.csv')['store_id']
STORES_IDS = list(STORES_IDS.unique())


#SPLITS for lags creation
SHIFT_DAY  = 28
N_LAGS     = 15
LAGS_SPLIT = [col for col in range(SHIFT_DAY,SHIFT_DAY+N_LAGS)]
ROLS_SPLIT = []
for i in [1,7,14]:
    for j in [7,14,30,60]:
        ROLS_SPLIT.append([i,j])

########################### Aux Models
# If you don't want to wait hours and hours
# to have result you can train each store 
# in separate kernel and then just join result.

# If we want to use pretrained models we can 
## skip training 
## (in our case do dummy training
##  to show that we are good with memory
##  and you can safely use this (all kernel) code)
if USE_AUX:
    lgb_params['n_estimators'] = 2
    
# Here is some 'logs' that can compare
#Train CA_1
#[100]	valid_0's rmse: 2.02289
#[200]	valid_0's rmse: 2.0017
#[300]	valid_0's rmse: 1.99239
#[400]	valid_0's rmse: 1.98471
#[500]	valid_0's rmse: 1.97923
#[600]	valid_0's rmse: 1.97284
#[700]	valid_0's rmse: 1.96763
#[800]	valid_0's rmse: 1.9624
#[900]	valid_0's rmse: 1.95673
#[1000]	valid_0's rmse: 1.95201
#[1100]	valid_0's rmse: 1.9476
#[1200]	valid_0's rmse: 1.9434
#[1300]	valid_0's rmse: 1.9392
#[1400]	valid_0's rmse: 1.93446

#Train CA_2
#[100]	valid_0's rmse: 1.88949
#[200]	valid_0's rmse: 1.84767
#[300]	valid_0's rmse: 1.83653
#[400]	valid_0's rmse: 1.82909
#[500]	valid_0's rmse: 1.82265
#[600]	valid_0's rmse: 1.81725
#[700]	valid_0's rmse: 1.81252
#[800]	valid_0's rmse: 1.80736
#[900]	valid_0's rmse: 1.80242
#[1000]	valid_0's rmse: 1.79821
#[1100]	valid_0's rmse: 1.794
#[1200]	valid_0's rmse: 1.78973
#[1300]	valid_0's rmse: 1.78552
#[1400]	valid_0's rmse: 1.78158

########################### Train Models
#################################################################################
import os
for store_id in STORES_IDS:
    print('Train', store_id)
    
    # Get grid for current store
    grid_df, features_columns = get_data_by_store(store_id)
    
    # Masks for 
    # Train (All data less than 1913)
    # "Validation" (Last 28 days - not real validatio set)
    # Test (All data greater than 1913 day, 
    #       with some gap for recursive features)
    train_mask = grid_df['d']<=END_TRAIN
    valid_mask = train_mask&(grid_df['d']>(END_TRAIN-P_HORIZON))
    preds_mask = grid_df['d']>(END_TRAIN-365)
    
    # Apply masks and save lgb dataset as bin
    # to reduce memory spikes during dtype convertations
    # https://github.com/Microsoft/LightGBM/issues/1032
    # "To avoid any conversions, you should always use np.float32"
    # or save to bin before start training
    # https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/53773
    train_data = lgb.Dataset(grid_df[train_mask][features_columns], 
                       label=grid_df[train_mask][TARGET])
    train_data.save_binary('train_data.bin')
    train_data = lgb.Dataset('train_data.bin')
    
    valid_data = lgb.Dataset(grid_df[valid_mask][features_columns], 
                       label=grid_df[valid_mask][TARGET])
    
    # Saving part of the dataset for later predictions
    # Removing features that we need to calculate recursively 
    grid_df = grid_df[preds_mask].reset_index(drop=True)
    keep_cols = [col for col in list(grid_df) if '_tmp_' not in col]
    grid_df = grid_df[keep_cols]
    grid_df.to_pickle('test_'+store_id+f'_d{END_TRAIN}_v2.pkl')
    del grid_df
    
    # Launch seeder again to make lgb training 100% deterministic
    # with each "code line" np.random "evolves" 
    # so we need (may want) to "reset" it
    seed_everything(SEED)
    estimator = lgb.train(lgb_params, train_data, valid_sets = [train_data, valid_data], verbose_eval = 100)
    
    # Save model - it's not real '.bin' but a pickle file
    # estimator = lgb.Booster(model_file='model.txt')
    # can only predict with the best iteration (or the saving iteration)
    # pickle.dump gives us more flexibility
    # like estimator.predict(TEST, num_iteration=100)
    # num_iteration - number of iteration want to predict with, 
    # NULL or <= 0 means use best iteration
    model_name = 'lgb_model_'+store_id+'_v'+str(END_TRAIN)+'.bin'
    pickle.dump(estimator, open(model_name, 'wb'))

    # Remove temporary files and objects 
    # to free some hdd space and ram memory
    #!rm train_data.bin
    os.remove('train_data.bin')
    del train_data, valid_data 
   # del estimator
    gc.collect()
    
    # "Keep" models features for predictions
    MODEL_FEATURES = features_columns

########################### Predict
#################################################################################

# Create Dummy DataFrame to store predictions
all_preds = pd.DataFrame()

# Join back the Test dataset with 
# a small part of the training data 
# to make recursive features
base_test = get_base_test()

# Timer to measure predictions time 
main_time = time.time()

grid_df, MODEL_FEATURES = get_data_by_store('CA_1')
del grid_df

# Loop over each prediction day
# As rolling lags are the most timeconsuming
# we will calculate it for whole day
for PREDICT_DAY in range(1, P_HORIZON + 1):    
    print('Predict | Day:', PREDICT_DAY)
    start_time = time.time()

    # Make temporary grid to calculate rolling lags
    grid_df = base_test.copy()
    grid_df = pd.concat([grid_df] + [make_lag_roll(x) for x in ROLS_SPLIT], axis=1)
 #   grid_df = pd.concat([grid_df, df_parallelize_run(make_lag_roll, ROLS_SPLIT)], axis=1)
        
    for store_id in STORES_IDS:
        # print('predicting',store_id)
        # Read all our models and make predictions
        # for each day/store pairs
        model_path = 'lgb_model_'+store_id+'_v'+str(END_TRAIN)+'.bin' 
        if USE_AUX:
            model_path = AUX_MODELS + model_path
        
        estimator = pickle.load(open(model_path, 'rb'))
        
        day_mask = (base_test['d']==(END_TRAIN+PREDICT_DAY))
        store_mask = base_test['store_id']==store_id
        
        mask = (day_mask)&(store_mask)
        base_test[TARGET][mask] = estimator.predict(grid_df[mask][MODEL_FEATURES])
    
    # Make good column naming and add 
    # to all_preds DataFrame
    temp_df = base_test[day_mask][['id',TARGET]]
    temp_df.columns = ['id','F'+str(PREDICT_DAY)]
    if 'id' in list(all_preds):
        all_preds = all_preds.merge(temp_df, on=['id'], how='left')
    else:
        all_preds = temp_df.copy()
        
    print('#'*10, ' %0.2f min round |' % ((time.time() - start_time) / 60),
                  ' %0.2f min total |' % ((time.time() - main_time) / 60),
                  ' %0.2f day sales |' % (temp_df['F'+str(PREDICT_DAY)].sum()))
    del temp_df
    
all_preds = all_preds.reset_index(drop=True)
all_preds

########################### Export
#################################################################################
# Reading competition sample submission and
# merging our predictions
# As we have predictions only for "_validation" data
# we need to do fillna() for "_evaluation" items
submission = pd.read_csv(ORIGINAL+'sample_submission.csv')[['id']]
submission = submission.merge(all_preds, on=['id'], how='left').fillna(0)
submission.loc[:,'F7'] *= 0.9
submission.loc[:,'F1':] *= 0.96
submission.to_csv('point_frc'+str(END_TRAIN)+'.csv', index=False)

# Summary

# Of course here is no magic at all.
# No "Novel" features and no brilliant ideas.
# We just carefully joined all
# our previous fe work and created a model.

# Also!
# In my opinion this strategy is a "dead end".
# Overfits a lot LB and with 1 final submission 
# you have no option to risk.


# Improvement should come from:
# Loss function
# Data representation
# Stable CV
# Good features reduction strategy
# Predictions stabilization with NN
# Trend prediction
# Real zero sales detection/classification


# Good kernels references 
## (the order is random and the list is not complete):
# https://www.kaggle.com/ragnar123/simple-lgbm-groupkfold-cv
# https://www.kaggle.com/jpmiller/grouping-items-by-stockout-pattern
# https://www.kaggle.com/headsortails/back-to-predict-the-future-interactive-m5-eda
# https://www.kaggle.com/sibmike/m5-out-of-stock-feature
# https://www.kaggle.com/mayer79/m5-forecast-attack-of-the-data-table
# https://www.kaggle.com/yassinealouini/seq2seq
# https://www.kaggle.com/kailex/m5-forecaster-v2
# https://www.kaggle.com/aerdem4/m5-lofo-importance-on-gpu-via-rapids-xgboost


# Features were created in these kernels:
## 
# Mean encodings and PCA options
# https://www.kaggle.com/kyakovlev/m5-custom-features
##
# Lags and rolling lags
# https://www.kaggle.com/kyakovlev/m5-lags-features
##
# Base Grid and base features (calendar/price/etc)
# https://www.kaggle.com/kyakovlev/m5-simple-fe


# Personal request
# Please don't upvote any ensemble and copypaste kernels
## The worst case is ensemble without any analyse.
## The best choice - just ignore it.
## I would like to see more kernels with interesting and original approaches.
## Don't feed copypasters with upvotes.

## It doesn't mean that you should not fork and improve others kernels
## but I would like to see params and code tuning based on some CV and analyse
## and not only on LB probing.
## Small changes could be shared in comments and authors can improve their kernel.

## Feel free to criticize this kernel as my knowlege is very limited
## and I can be wrong in code and descriptions. 
## Thank you.

