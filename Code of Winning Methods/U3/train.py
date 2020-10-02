#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# General imports
import numpy as np
import pandas as pd
import sys, gc, time, warnings, pickle, random, psutil

# custom imports
from multiprocessing import Pool        # Multiprocess Runs

import warnings
warnings.filterwarnings('ignore')

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tqdm
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler #StandardScaler

from keras_models import create_model17EN1EN2emb1, create_model17noEN1EN2, create_model17

import json
with open('SETTINGS.json', 'r') as myfile:
    datafile=myfile.read()
SETTINGS = json.loads(datafile)


data_path = SETTINGS['RAW_DATA_DIR']
PROCESSED_DATA_DIR = SETTINGS['PROCESSED_DATA_DIR']
MODELS_DIR = SETTINGS['MODELS_DIR']
LGBM_DATASETS_DIR = SETTINGS['LGBM_DATASETS_DIR']

###############################################################################
################################# LIGHTGBM ####################################
###############################################################################



########################### Helpers
###############################################################################
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


########################### Helper to load data by store ID
###############################################################################
# Read data
lag_features=[
       'sales_lag_28', 'sales_lag_29', 'sales_lag_30', 'sales_lag_31', 
       'sales_lag_32', 'sales_lag_33', 'sales_lag_34', 'sales_lag_35', 
       'sales_lag_36', 'sales_lag_37', 'sales_lag_38',
       'sales_lag_39', 'sales_lag_40', 'sales_lag_41', 'sales_lag_42',
       'rolling_mean_7', 'rolling_mean_14', 
       'rolling_mean_30', 'rolling_std_30', 'rolling_mean_60',
       'rolling_std_60',  'rolling_mean_180', 'rolling_std_180',
       'rolling_mean_tmp_1_7', 'rolling_mean_tmp_1_14',
       'rolling_mean_tmp_1_30', 'rolling_mean_tmp_1_60',
       'rolling_mean_tmp_7_7', 'rolling_mean_tmp_7_14',
       'rolling_mean_tmp_7_30', 'rolling_mean_tmp_7_60',
       'rolling_mean_tmp_14_7', 'rolling_mean_tmp_14_14',
       'rolling_mean_tmp_14_30', 'rolling_mean_tmp_14_60'
    ]

def get_data_by_store(store):
    
    # Read and contact basic feature
    df = pd.concat([pd.read_pickle(BASE),
                    pd.read_pickle(PRICE).iloc[:,2:],
                    pd.read_pickle(CALENDAR).iloc[:,2:]],
                    axis=1)
    
    # Leave only relevant store
    df = df[df['store_id']==store]

    
    df2 = pd.read_pickle(MEAN_ENC)[mean_features]
    df2 = df2[df2.index.isin(df.index)]
    
#     df3 = pd.read_pickle(LAGS).iloc[:,3:]
    df3 = pd.read_pickle(LAGS)[lag_features]
    df3 = df3[df3.index.isin(df.index)]
    
    df = pd.concat([df, df2], axis=1)
    del df2 # to not reach memory limit 
    
    df = pd.concat([df, df3], axis=1)
    del df3 # to not reach memory limit 
    
    # Create features list
    features = [col for col in list(df) if col not in remove_features]
    df = df[['id','d',TARGET]+features]
    
    # Skipping first n rows
    df = df[df['d']>=START_TRAIN].reset_index(drop=True)
    
    return df, features

# Recombine Test set after training
def get_base_test():
    base_test = pd.DataFrame()

    for store_id in STORES_IDS:
        temp_df = pd.read_pickle('lgbtrainings/test_'+store_id+'.pkl')
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
###############################################################################
import lightgbm as lgb
lgb_params = {
                    'boosting_type': 'gbdt',
                    'objective': 'tweedie',
                    'tweedie_variance_power': 1.1,
                    'metric': 'rmse',
                    'subsample': 0.6,
                    'subsample_freq': 1,
                    'learning_rate': 0.02,
                    'num_leaves': 2**11-1,
                    'min_data_in_leaf': 2**12-1,
                    'feature_fraction': 0.6,
                    'max_bin': 100,
                    'n_estimators': 1600,
                    'boost_from_average': False,
                    'verbose': -1,
                    'num_threads': 12
                } 


########################### Vars
###############################################################################

# Our model version
SEED = 42                        # We want all things
seed_everything(SEED)            # to be as deterministic 
lgb_params['seed'] = SEED        # as possible
N_CORES = psutil.cpu_count()     # Available CPU cores
#N_CORES=7 #from 24


#LIMITS and const
TARGET      = 'sales'            # Our target
P_HORIZON   = 28                 # Prediction horizon
USE_AUX     = False               # Use or not pretrained models

#FEATURES to remove
remove_features = ['id','state_id','store_id',
                   'date','wm_yr_wk','d',TARGET]

mean_features   = ['enc_cat_id_mean','enc_cat_id_std',
                   'enc_dept_id_mean','enc_dept_id_std',
                   'enc_item_id_mean','enc_item_id_std'] 

#PATHS for Features
ORIGINAL = data_path#
BASE     = PROCESSED_DATA_DIR+'grid_part_1_eval.pkl'
PRICE    = PROCESSED_DATA_DIR+'grid_part_2_eval.pkl'
CALENDAR = PROCESSED_DATA_DIR+'grid_part_3_eval.pkl'
LAGS     = PROCESSED_DATA_DIR+'lags_df_28_eval.pkl'
MEAN_ENC = PROCESSED_DATA_DIR+'mean_encoding_df_eval.pkl'


# AUX(pretrained) Models paths
#AUX_MODELS = data_path+'m5-aux-models/'


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
        

#N_CORES=7 #from 24


#dataPath = '/var/data/m5-forecasting-accuracy/'
# train_df = pd.read_csv(dataPath+'sales_train_validation.csv')
train_df = pd.read_csv(data_path+'sales_train_evaluation.csv')
calendar1 = pd.read_csv(data_path+'calendar.csv')
sell_prices1 = pd.read_csv(data_path+'sell_prices.csv')

VER =  '4' 
START_TRAIN = 0 #1186                  # We can skip some rows (Nans/faster training)
END_TRAIN   = 1941 


grid_df, features_columns = get_data_by_store(STORES_IDS[0])



rounds_per_store1={
     'CA_1': 700,
     'CA_2': 1100,
     'CA_3': 1600,
     'CA_4': 1500,
     'TX_1': 1000, 
     'TX_2': 1000,
     'TX_3': 1000,
     'WI_1': 1600,
     'WI_2': 1500,
     'WI_3': 1100
}
#rounds_per_store1={
#     'CA_1': 1,
#     'CA_2': 1,
#     'CA_3': 1,
#     'CA_4': 1,
#     'TX_1': 1,
#     'TX_2': 1,
#     'TX_3': 1,
#     'WI_1': 1,
#     'WI_2': 1,
#     'WI_3': 1
#}
########################### Train Models
###############################################################################
for store_id in STORES_IDS:
    print('Train', store_id)
    lgb_params['n_estimators'] = rounds_per_store1[store_id]
    
    # Get grid for current store
    grid_df, features_columns = get_data_by_store(store_id)


    train_mask = grid_df['d']<=END_TRAIN
#     valid_mask = (grid_df['d']>END_TRAIN)&(grid_df['d']<=END_TRAIN+28)
    valid_mask = train_mask&(grid_df['d']>(END_TRAIN-P_HORIZON)) #pseudovalidation
    preds_mask = grid_df['d']>(END_TRAIN-100)

    # Apply masks and save lgb dataset as bin
    # to reduce memory spikes during dtype convertations
    # https://github.com/Microsoft/LightGBM/issues/1032
    # "To avoid any conversions, you should always use np.float32"
    # or save to bin before start training
    # https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/53773
    train_data = lgb.Dataset(grid_df[train_mask][features_columns], 
                       label=grid_df[train_mask][TARGET])
    train_data.save_binary(LGBM_DATASETS_DIR+'train_data.bin')
    train_data = lgb.Dataset(LGBM_DATASETS_DIR+'train_data.bin')

    valid_data = lgb.Dataset(grid_df[valid_mask][features_columns], 
                       label=grid_df[valid_mask][TARGET])

    # Saving part of the dataset for later predictions
    # Removing features that we need to calculate recursively 
    grid_df = grid_df[preds_mask].reset_index(drop=True)
    keep_cols = [col for col in list(grid_df) if '_tmp_' not in col]
    grid_df = grid_df[keep_cols]
#     grid_df.to_pickle('lgbtrainings/test_'+store_id+'MPh.pkl')
    grid_df.to_pickle(LGBM_DATASETS_DIR+'test_'+store_id+'.pkl')
#     grid_df.to_pickle('lgbtrainings/test_'+'END_TRAIN'+str(END_TRAIN)+'store_id'+store_id+'.pkl')
    del grid_df

    # Launch seeder again to make lgb training 100% deterministic
    # with each "code line" np.random "evolves" 
    # so we need (may want) to "reset" it
    seed_everything(SEED)
    estimator = lgb.train(lgb_params,
                          train_data,
                          valid_sets = [valid_data],
                          verbose_eval = 100,
                          )

    # Save model - it's not real '.bin' but a pickle file
    # estimator = lgb.Booster(model_file='model.txt')
    # can only predict with the best iteration (or the saving iteration)
    # pickle.dump gives us more flexibility
    # like estimator.predict(TEST, num_iteration=100)
    # num_iteration - number of iteration want to predict with, 
    # NULL or <= 0 means use best iteration

#     model_name = 'lgbweights/lgb_model_'+store_id+'END_TRAIN'+str(END_TRAIN)+'_v'+str(VER)+'.bin'
    model_name = MODELS_DIR+'lgbm_finalmodel_'+store_id+'_v'+str(VER)+'.bin'

    pickle.dump(estimator, open(model_name, 'wb'))

    # Remove temporary files and objects 
    # to free some hdd space and ram memory
#    !rm lgbtrainings/train_data.bin
    os.remove(LGBM_DATASETS_DIR+"train_data.bin")
    del train_data, valid_data, estimator
    gc.collect()

# "Keep" models features for predictions
MODEL_FEATURES = features_columns

with open(LGBM_DATASETS_DIR+'lgbm_features.txt', 'w') as filehandle:
    for listitem in MODEL_FEATURES:
        filehandle.write('%s\n' % listitem)


###############################################################################
################################# KERAS #######################################
###############################################################################
        
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

def prep_calendar(df):
    df = df.drop(["date", "weekday"], axis=1)
    df = df.assign(d = df.d.str[2:].astype(int))
    df = df.fillna("missing")
    cols = list(set(df.columns) - {"wm_yr_wk", "d"})
    df[cols] = OrdinalEncoder(dtype="int").fit_transform(df[cols])
    df = reduce_mem_usage(df)
    return df

def prep_selling_prices(df):
    gr = df.groupby(["store_id", "item_id"])["sell_price"]
    df["sell_price_rel_diff"] = gr.pct_change()
    df["sell_price_roll_sd7"] = gr.transform(lambda x: x.rolling(7).std())
    df["sell_price_cumrel"] = (gr.shift(0) - gr.cummin()) / (1 + gr.cummax() - gr.cummin())
    df = reduce_mem_usage(df)
    return df

def reshape_sales(df, drop_d = None):
    if drop_d is not None:
        df = df.drop(["d_" + str(i + 1) for i in range(drop_d)], axis=1)
    df = df.assign(id=df.id.str.replace("_validation", ""))
    df = df.reindex(columns=df.columns.tolist() + ["d_" + str(1941 + i + 1) for i in range(1 * 28)])
    df = df.melt(id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
                 var_name='d', value_name='demand')
    df = df.assign(d=df.d.str[2:].astype("int16"))
    return df

def prep_sales(df):
    
    df['lag_t28'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(28))
    df['rolling_mean_28_7'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(7).mean())
    df['rolling_mean_28_28'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(28).mean())
    
    df['rolling_median_28_7'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(7).median())
    df['rolling_median_28_28'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(28).median()) 
    df = reduce_mem_usage(df)

    return df


def preprocess_data(X, scaler=None):
    if not scaler:
        scaler = MinMaxScaler((0,1))
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler


# Read raw data
calendar = pd.read_csv(data_path+ "calendar.csv")
selling_prices = pd.read_csv(data_path+ "sell_prices.csv")
sample_submission = pd.read_csv(data_path+ "sample_submission.csv")
sales = pd.read_csv(data_path+ "sales_train_evaluation.csv")
# sales = pd.read_csv(data_path+"sales_train_validation.csv")

# Prepare data for keras
calendar = prep_calendar(calendar)
selling_prices = prep_selling_prices(selling_prices)
# sales = reshape_sales(sales, 0)
sales = reshape_sales(sales, 1000)
sales = sales.merge(calendar, how="left", on="d")
gc.collect()
#sales.head()

sales = sales.merge(selling_prices, how="left", on=["wm_yr_wk", "store_id", "item_id"])
sales.drop(["wm_yr_wk"], axis=1, inplace=True)
gc.collect()
sales = prep_sales(sales)

del selling_prices


cat_id_cols = ["item_id", "dept_id", "store_id", "cat_id", "state_id"]
cat_cols = cat_id_cols + ["wday", "month", "year", "event_name_1", 
                          "event_type_1", "event_name_2", "event_type_2"]

# Encode data with loop to minimize memory use
for i, v in tqdm.tqdm(enumerate(cat_id_cols)):
    sales[v] = OrdinalEncoder(dtype="int").fit_transform(sales[[v]])

sales = reduce_mem_usage(sales)
#sales.head()
gc.collect()

# add feature
sales['logd']=np.log1p(sales.d-sales.d.min())



num_cols = [  "sell_price", 
              "sell_price_rel_diff",
              "rolling_mean_28_7",
              "rolling_mean_28_28",             
              "rolling_median_28_7",
              "rolling_median_28_28", 
              "logd"

           ]
bool_cols = ["snap_CA", "snap_TX", "snap_WI"]
dense_cols = num_cols + bool_cols

# Need to do column by column due to memory constraints
for i, v in tqdm.tqdm(enumerate(num_cols)):
    sales[v] = sales[v].fillna(sales[v].median())##############################
    

gc.collect()


# Input dict for training with a dense array and separate inputs for each embedding input
def make_X(df):
    X = {"dense1": df[dense_cols].values}
#     X = {"dense1": df[dense_cols].to_numpy()}
    for i, v in enumerate(cat_cols):
#         X[v] = df[[v]].to_numpy()
        X[v] = df[[v]].values
    return X


END_TRAIN=1941
et=1941


base_epochs=30
ver='EN1EN2Emb1'


flag = (sales.d < et+1 )& (sales.d > et+1-17*28 )

X_train = make_X(sales[flag])
y_train = sales["demand"][flag].values

X_train['dense1'], scaler1 = preprocess_data(X_train['dense1'])

val_long = sales[(sales.d >= et+1)].copy()

#train
model = create_model17EN1EN2emb1(num_dense_features=len(dense_cols), lr=0.0001)

history = model.fit(X_train,  y_train,
                        batch_size=2 ** 14,
                        epochs=base_epochs,
                        shuffle=True,
                        verbose = 2
                   )

model.save_weights(MODELS_DIR+'Keras_CatEmb_final3_et'+str(et)+'ep'+str(base_epochs)+'_ver-'+ver+'.h5')     
#import matplotlib.pyplot as plt 
#plt.plot(history.history['loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.savefig('plt_hist'+str(et)+ver)
#plt.show()

for j in range(4):        
    model.fit(X_train, y_train,
                        batch_size=2 ** 14,
                        epochs=1,
                        shuffle=True,
                        verbose = 2
             )
    model.save_weights(MODELS_DIR+'Keras_CatEmb_final3_et'+str(et)+'ep'+str(base_epochs+1+j)+'_ver-'+ver+'.h5')          



# Second Model
base_epochs=30
ver='noEN1EN2'

#train
model = create_model17noEN1EN2(num_dense_features=len(dense_cols), lr=0.0001)
history = model.fit(X_train,  y_train,
                        batch_size=2 ** 14,
                        epochs=base_epochs,
                        shuffle=True,
                        verbose = 2
                   )

model.save_weights(MODELS_DIR+'Keras_CatEmb_final3_et'+str(et)+'ep'+str(base_epochs)+'_ver-'+ver+'.h5')      
#plt.plot(history.history['loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.savefig('plt_hist'+str(et)+ver)
#plt.show()
for j in range(4):        
    model.fit(X_train, y_train,
                        batch_size=2 ** 14,
                        epochs=1,
                        shuffle=True,
                        verbose = 2
             )
    model.save_weights(MODELS_DIR+'Keras_CatEmb_final3_et'+str(et)+'ep'+str(base_epochs+1+j)+'_ver-'+ver+'.h5')          

# del model
gc.collect()

# Third Model
base_epochs=30
ver='17last'

val_long = sales[(sales.d >= et+1)].copy()

#train
model = create_model17(num_dense_features=len(dense_cols), lr=0.0001)
history = model.fit(X_train,  y_train,
                        batch_size=2 ** 14,
                        epochs=base_epochs,
                        shuffle=True,
                        verbose = 2
                   )

model.save_weights(MODELS_DIR+'Keras_CatEmb_final3_et'+str(et)+'ep'+str(base_epochs)+'_ver-'+ver+'.h5')      
#plt.plot(history.history['loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.savefig('plt_hist'+str(et)+ver)
#plt.show()
for j in range(4):        
    model.fit(X_train, y_train,
                        batch_size=2 ** 14,
                        epochs=1,
                        shuffle=True,
                        verbose = 2
             )
    model.save_weights(MODELS_DIR+'Keras_CatEmb_final3_et'+str(et)+'ep'+str(base_epochs+1+j)+'_ver-'+ver+'.h5')          


# del model
gc.collect()        