#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# General imports
import numpy as np
import pandas as pd
import  sys, gc, time, warnings, pickle, random, psutil

# custom imports
from multiprocessing import Pool        # Multiprocess Runs

import warnings
warnings.filterwarnings('ignore')


import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tqdm
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler #StandardScaler

from keras_models import create_model17EN1EN2emb1, create_model17noEN1EN2, create_model17

import scipy.stats  as stats

import json
with open('SETTINGS.json', 'r') as myfile:
    datafile=myfile.read()
SETTINGS = json.loads(datafile)


data_path = SETTINGS['RAW_DATA_DIR']
PROCESSED_DATA_DIR = SETTINGS['PROCESSED_DATA_DIR']
MODELS_DIR = SETTINGS['MODELS_DIR']
LGBM_DATASETS_DIR = SETTINGS['LGBM_DATASETS_DIR']
SUBMISSION_DIR = SETTINGS['SUBMISSION_DIR']


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
        temp_df = pd.read_pickle(LGBM_DATASETS_DIR+'test_'+store_id+'.pkl')
        temp_df['store_id'] = store_id
        base_test = pd.concat([base_test, temp_df]).reset_index(drop=True)
    
    return base_test


########################### Helper to make dynamic rolling lags
###############################################################################
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


# Our model version
SEED = 42                        # We want all things
seed_everything(SEED)            # to be as deterministic 
#lgb_params['seed'] = SEED        # as possible
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
ORIGINAL = data_path#+'m5-forecasting-accuracy/'
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
        


# train_df = pd.read_csv(dataPath+'sales_train_validation.csv')
train_df = pd.read_csv(data_path+'sales_train_evaluation.csv')
calendar1 = pd.read_csv(data_path+'calendar.csv')
sell_prices1 = pd.read_csv(data_path+'sell_prices.csv')

VER =  '4' 
START_TRAIN = 0 #1186                  # We can skip some rows (Nans/faster training)
END_TRAIN   = 1941 


#grid_df, features_columns = get_data_by_store(STORES_IDS[0])

 # 
MODEL_FEATURES = []

 # open file and read the content in a list
with open(LGBM_DATASETS_DIR+'lgbm_features.txt', 'r') as filehandle:
    for line in filehandle:
        # remove linebreak which is the last character of the string
        currentPlace = line[:-1]
        # add item to the list
        MODEL_FEATURES.append(currentPlace)




########################### Predict
###############################################################################

# Create Dummy DataFrame to store predictions
all_preds = pd.DataFrame()

# Join back the Test dataset with 
# a small part of the training data 
# to make recursive features
# base_test = get_base_test_et(END_TRAIN)
base_test = get_base_test()
# base_test = get_base_test_MPh()

# Timer to measure predictions time 
main_time = time.time()

# Loop over each prediction day
# As rolling lags are the most timeconsuming
# we will calculate it for whole day
for PREDICT_DAY in range(1,29):    
    print('Predict | Day:', PREDICT_DAY)
    start_time = time.time()

    # Make temporary grid to calculate rolling lags
    grid_df = base_test.copy()
    grid_df = pd.concat([grid_df, df_parallelize_run(make_lag_roll, ROLS_SPLIT)], axis=1)

    for store_id in STORES_IDS:

        # Read all our models and make predictions
        model_path = MODELS_DIR+'lgbm_finalmodel_'+store_id+'_v'+str(VER)+'.bin'

        estimator = pickle.load(open(model_path, 'rb'))

        day_mask = base_test['d']==(END_TRAIN+PREDICT_DAY)
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

#### SAVE TO DISK
valDF = pd.read_csv(ORIGINAL+'sample_submission.csv')[['id']]
valDF = valDF.merge(all_preds, on=['id'], how='left').fillna(0)
# valDF.iloc[:,30490:]=valDF.iloc[:,:30490].values
valDF.to_csv(SUBMISSION_DIR+'lgbm_final_VER'+str(VER)+'.csv.gz', 
             index=False,compression='gzip')
val_predsDF_lgbm=valDF.copy()
gc.collect()


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

#read raw data
calendar = pd.read_csv(data_path+ "calendar.csv")
selling_prices = pd.read_csv(data_path+ "sell_prices.csv")
sample_submission = pd.read_csv(data_path+ "sample_submission.csv")
sales = pd.read_csv(data_path+ "sales_train_evaluation.csv")
# sales = pd.read_csv(data_path+ "sales_train_validation.csv")


# prepare data for keras
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



# In loop to minimize memory use
for i, v in tqdm.tqdm(enumerate(cat_id_cols)):
    sales[v] = OrdinalEncoder(dtype="int").fit_transform(sales[[v]])

sales = reduce_mem_usage(sales)
#sales.head()
gc.collect()

# add feature
sales['logd']=np.log1p(sales.d-sales.d.min())

# numerical cols
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
    sales[v] = sales[v].fillna(sales[v].median())################################################
    

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

# Make train data to use num cols for scaling test data
flag = (sales.d < et+1 )& (sales.d > et+1-17*28 )
X_train = make_X(sales[flag])
X_train['dense1'], scaler1 = preprocess_data(X_train['dense1'])


#predict model 1
base_epochs=30
ver='EN1EN2Emb1'
val_long = sales[(sales.d >= et+1)].copy()

model = create_model17EN1EN2emb1(num_dense_features=len(dense_cols), lr=0.0001)
for i in range(et+1, et+1+28):    
    forecast = make_X(val_long[val_long.d == i])
#         forecast['dense1'] =  forecast['dense1'][:,[x for x in range(forecast['dense1'].shape[1]) if x not in exclude_dense]]
    forecast['dense1'], scaler = preprocess_data(forecast['dense1'], scaler1)

    model.load_weights(MODELS_DIR+'Keras_CatEmb_final3_et'+str(et)+'ep'+str(base_epochs)+'_ver-'+ver+'.h5')  
    pred = model.predict(forecast, batch_size=2 ** 14)
    for j in range(1,5):
        model.load_weights(MODELS_DIR+'Keras_CatEmb_final3_et'+str(et)+'ep'+str(base_epochs+j)+'_ver-'+ver+'.h5')
        pred += model.predict(forecast, batch_size=2 ** 14)
    pred /= 5

    val_long.loc[val_long.d == i, "demand"] = pred.clip(0) #* 1.02

val_preds = val_long.demand[(val_long.d >= et+1)&(val_long.d < et+1+28)]
val_preds=val_preds.values.reshape(30490,28, order='F')

#write to disk
ss = pd.read_csv(data_path+'sample_submission.csv')[['id']]
val_predsDF = pd.DataFrame(val_preds,columns=['F'+str(i) for i in range(1,29)])
val_predsDF=pd.concat([ss.id, pd.concat([pd.DataFrame(np.zeros((30490,28),dtype=np.float32),columns=['F'+str(i) for i in range(1,29)]),
           val_predsDF],0).reset_index(drop=True) ],1)
val_predsDF.to_csv(SUBMISSION_DIR+'Keras_CatEmb_final3_et'+str(et)+'_ver-'+ver+'.csv.gz',index=False,compression='gzip')
val_predsDF1=val_predsDF.copy()

# del model
gc.collect()

#predict model 2
base_epochs=30
ver='noEN1EN2'
val_long = sales[(sales.d >= et+1)].copy()

model = create_model17noEN1EN2(num_dense_features=len(dense_cols), lr=0.0001)
for i in range(et+1, et+1+28):    
    forecast = make_X(val_long[val_long.d == i])
#         forecast['dense1'] =  forecast['dense1'][:,[x for x in range(forecast['dense1'].shape[1]) if x not in exclude_dense]]
    forecast['dense1'], scaler = preprocess_data(forecast['dense1'], scaler1)

    model.load_weights(MODELS_DIR+'Keras_CatEmb_final3_et'+str(et)+'ep'+str(base_epochs)+'_ver-'+ver+'.h5')  
    pred = model.predict(forecast, batch_size=2 ** 14)
    for j in range(1,5):
        model.load_weights(MODELS_DIR+'Keras_CatEmb_final3_et'+str(et)+'ep'+str(base_epochs+j)+'_ver-'+ver+'.h5')
        pred += model.predict(forecast, batch_size=2 ** 14)
    pred /= 5

    val_long.loc[val_long.d == i, "demand"] = pred.clip(0) #* 1.02

val_preds = val_long.demand[(val_long.d >= et+1)&(val_long.d < et+1+28)]
val_preds=val_preds.values.reshape(30490,28, order='F')

#write to disk
ss = pd.read_csv(data_path+'sample_submission.csv')[['id']]
val_predsDF = pd.DataFrame(val_preds,columns=['F'+str(i) for i in range(1,29)])
val_predsDF=pd.concat([ss.id, pd.concat([pd.DataFrame(np.zeros((30490,28),dtype=np.float32),columns=['F'+str(i) for i in range(1,29)]),
           val_predsDF],0).reset_index(drop=True) ],1)
val_predsDF.to_csv(SUBMISSION_DIR+'Keras_CatEmb_final3_et'+str(et)+'_ver-'+ver+'.csv.gz',index=False,compression='gzip')
val_predsDF2=val_predsDF.copy()

#predict model 3
base_epochs=30
ver='17last'
val_long = sales[(sales.d >= et+1)].copy()

model = create_model17(num_dense_features=len(dense_cols), lr=0.0001)
for i in range(et+1, et+1+28):    
    forecast = make_X(val_long[val_long.d == i])
#         forecast['dense1'] =  forecast['dense1'][:,[x for x in range(forecast['dense1'].shape[1]) if x not in exclude_dense]]
    forecast['dense1'], scaler = preprocess_data(forecast['dense1'], scaler1)

    model.load_weights(MODELS_DIR+'Keras_CatEmb_final3_et'+str(et)+'ep'+str(base_epochs)+'_ver-'+ver+'.h5')  
    pred = model.predict(forecast, batch_size=2 ** 14)
    for j in range(1,5):
        model.load_weights(MODELS_DIR+'Keras_CatEmb_final3_et'+str(et)+'ep'+str(base_epochs+j)+'_ver-'+ver+'.h5')
        pred += model.predict(forecast, batch_size=2 ** 14)
    pred /= 5

    val_long.loc[val_long.d == i, "demand"] = pred.clip(0) #* 1.02

val_preds = val_long.demand[(val_long.d >= et+1)&(val_long.d < et+1+28)]
val_preds=val_preds.values.reshape(30490,28, order='F')

#write to disk
ss = pd.read_csv(data_path+'sample_submission.csv')[['id']]
val_predsDF = pd.DataFrame(val_preds,columns=['F'+str(i) for i in range(1,29)])
val_predsDF=pd.concat([ss.id, pd.concat([pd.DataFrame(np.zeros((30490,28),dtype=np.float32),columns=['F'+str(i) for i in range(1,29)]),
           val_predsDF],0).reset_index(drop=True) ],1)
val_predsDF.to_csv(SUBMISSION_DIR+'Keras_CatEmb_final3_et'+str(et)+'_ver-'+ver+'.csv.gz',index=False,compression='gzip')
val_predsDF3=val_predsDF.copy()


ver='avg3'
val_predsDF.iloc[:,1:] = (val_predsDF1.iloc[:,1:].values + val_predsDF2.iloc[:,1:].values + val_predsDF3.iloc[:,1:].values)/3
val_predsDF.to_csv(SUBMISSION_DIR+'Keras_CatEmb_final3_et'+str(et)+'_ver-'+ver+'.csv.gz',index=False,compression='gzip')




###############################################################################
################################# ENSEMBLE ####################################
###############################################################################

# Submissions for M5 accuracy competition, used as starting point to M5 uncertainty
final_preds_acc = val_predsDF_lgbm.copy()
final_preds_acc.iloc[:,1:] = (val_predsDF_lgbm.iloc[:,1:]**3 * val_predsDF.iloc[:,1:]) ** (1/4)

sales = pd.read_csv(data_path+ "sales_train_evaluation.csv")
period=28*26
m=np.mean(sales.iloc[:,-period:].values,0).mean()
#mv=np.mean(sales.iloc[:,-period:].values,0).max()
s=np.mean(sales.iloc[:,-period:].values,0).std()
mkd=np.mean(val_predsDF.iloc[30490:,1:],0).values
keras_outlier_days=np.where(mkd>m+6*s)[0]

# For days keras preds fails, replace with lgbm preds
for d in keras_outlier_days: 
    final_preds_acc.iloc[:,d+1]=val_predsDF_lgbm.iloc[:,d+1].values
#final_preds_acc.iloc[:,-1] = val_predsDF_lgbm.iloc[:,-1].values
final_preds_acc.to_csv(SUBMISSION_DIR+'lgbm3keras1.csv.gz',index=False,compression='gzip')


