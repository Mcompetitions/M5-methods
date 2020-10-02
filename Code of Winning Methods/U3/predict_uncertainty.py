#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###############################################################################
################################# M5 UNCERTAINTY ##############################
###############################################################################
import pandas as pd, numpy as np
#from matplotlib import pyplot as plt

import scipy.stats  as stats

import sys
acc_filename=sys.argv[1]
print('creating uncertainty predictions for: ' + acc_filename)

import json
with open('SETTINGS.json', 'r') as myfile:
    datafile=myfile.read()
SETTINGS = json.loads(datafile)

data_path = SETTINGS['RAW_DATA_DIR']
PROCESSED_DATA_DIR = SETTINGS['PROCESSED_DATA_DIR']
MODELS_DIR = SETTINGS['MODELS_DIR']
LGBM_DATASETS_DIR = SETTINGS['LGBM_DATASETS_DIR']
SUBMISSION_DIR = SETTINGS['SUBMISSION_DIR']


best = pd.read_csv(SUBMISSION_DIR + acc_filename)


# copy accuracy's private LB data to public LB data
# both public and private LB predictions will be equal, only private LB will be right
best.iloc[:30490,1:]=best.iloc[30490:,1:].values

# best.iloc[:,1:]=best.iloc[:,1:]*0.97 magical coef for private lb


# Exponential weighted Mean
c=0.04
best.iloc[:,1:]=best.iloc[:,1:].ewm(com=c,axis=1).mean().values


# read sales raw data
#sales = pd.read_csv(data_path+"sales_train_validation.csv")
#salesids=sales.id.values
#sales = pd.read_csv(data_path+"sales_train_evaluation.csv")
#sales.id=salesids
sales = pd.read_csv(data_path+"sales_train_evaluation.csv")
sales=sales.assign(id=sales.id.str.replace("_evaluation", "_validation"))


sub = best.merge(sales[["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]], on = "id")
sub["_all_"] = "Total"




# ## Different ratios for different aggregation levels
# 
# The higher the aggregation level, the more confident we are in the point 
#    prediction --> lower coef, relatively smaller range of quantiles


qs = np.array([0.005,0.025,0.165,0.25, 0.5, 0.75, 0.835, 0.975, 0.995])


def get_ratios(coef=0.15):
    qs2 = np.log(qs/(1-qs))*coef
    ratios = stats.norm.cdf(qs2)
    ratios /= ratios[4]
    ratios[-1] *= 1.03
    ratios = pd.Series(ratios, index=qs)
    return ratios.round(3)

def get_ratios2(coef=0.15, a=1.2):
    qs2 = np.log(qs/(1-qs))*coef
    ratios = stats.skewnorm.cdf(qs2, a)
    ratios /= ratios[4]
    ratios[-1] *= 1.02
    ratios = pd.Series(ratios, index=qs)
    return ratios.round(3)

def get_ratios3(coef=0.15, c=0.5, s=0.1):
    qs2 = qs*coef
    ratios = stats.powerlognorm.ppf(qs2, c, s)
    ratios /= ratios[4]
    ratios[0] *= 0.25
    ratios[-1] *= 1.02
    ratios = pd.Series(ratios, index=qs)
    return ratios.round(3)


def widen(array, pc):
    #array : array of df
    #pc: per cent (0:100)
    array[array<1]=array[array<1] * (1 - pc/100)
    array[array>1]=array[array>1] * (1 + pc/100)
    return array

level_coef_dict = {"id": widen((get_ratios2(coef=0.3)+get_ratios3(coef=.3, c=0.04, s=0.9))/2, pc=0.5), "item_id": widen(get_ratios2(coef=0.18, a=0.4),pc=0.5),
                   "dept_id": widen(get_ratios(coef=0.04),0.5), "cat_id": widen(get_ratios(coef=0.03),0.5),
                   "store_id": widen(get_ratios(coef=0.035),0.5), "state_id": widen(get_ratios(coef=0.03),0.5), 
                   "_all_": widen(get_ratios(coef=0.025),0.5),
                   ("state_id", "item_id"): widen(get_ratios2(coef=0.21, a=0.75), pc=0.5),  ("state_id", "dept_id"): widen(get_ratios(coef=0.05),0.5),
                    ("store_id","dept_id") : widen(get_ratios(coef=0.07),0.5), ("state_id", "cat_id"): widen(get_ratios(coef=0.04),0.5),
                    ("store_id","cat_id"): widen(get_ratios(coef=0.055),0.5)
                  }


def quantile_coefs(q, level):
    ratios = level_coef_dict[level]
               
    return ratios.loc[q].values

def get_group_preds(pred, level):
    df = pred.groupby(level)[cols].sum()
    q = np.repeat(qs, len(df))
#     q = np.repeat(np.expand_dims(qs,-1), len(df), -1)
#     print(q.shape)
    df = pd.concat([df]*9, axis=0, sort=False)
    df.reset_index(inplace = True)
#     print(quantile_coefs(q, level)[:, None].shape)
#     df[cols] *= quantile_coefs(q, level)[:, None]
    df[cols] *= np.repeat((quantile_coefs(q, level)[:, None]), len(cols),-1)
    if level != "id":
        df["id"] = [f"{lev}_X_{q:.3f}_validation" for lev, q in zip(df[level].values, q)]
    else:
        df["id"] = [f"{lev.replace('_validation', '')}_{q:.3f}_validation" for lev, q in zip(df[level].values, q)]
    df = df[["id"]+list(cols)]
    return df

def get_couple_group_preds(pred, level1, level2):
    df = pred.groupby([level1, level2])[cols].sum()
    q = np.repeat(qs, len(df))
    df = pd.concat([df]*9, axis=0, sort=False)
    df.reset_index(inplace = True)
#     df[cols] *= quantile_coefs(q, (level1, level2))[:, None]
    df[cols] *= np.repeat((quantile_coefs(q, (level1, level2))[:, None]), len(cols),-1)
    df["id"] = [f"{lev1}_{lev2}_{q:.3f}_validation" for lev1,lev2, q in 
                zip(df[level1].values,df[level2].values, q)]
    df = df[["id"]+list(cols)]
    return df

levels = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "_all_"]
couples = [("state_id", "item_id"),  ("state_id", "dept_id"),("store_id","dept_id"),
                            ("state_id", "cat_id"),("store_id","cat_id")]
cols = [f"F{i}" for i in range(1, 29)]

# Make predictions
df = []
for level in levels :
    df.append(get_group_preds(sub, level))
for level1,level2 in couples:
    df.append(get_couple_group_preds(sub, level1, level2))
df = pd.concat(df, axis=0, sort=False)
df.reset_index(drop=True, inplace=True)
df = pd.concat([df,df] , axis=0, sort=False)
df.reset_index(drop=True, inplace=True)
df.loc[df.index >= len(df.index)//2, "id"] = df.loc[df.index >= len(df.index)//2, "id"].str.replace(
                                    "_validation$", "_evaluation")

# salesGitem=sales.groupby('item_id')[['item_id']+[f"d_{i}" for i in range(914, 1914)]].sum()
#salesGitem=sales.groupby('item_id')[['item_id']+[f"d_{i}" for i in range(942, 1942)]].sum()

# Statistical computatition to help overwrite Level12
sales.sort_values('id',inplace=True)
sales.reset_index(drop=True, inplace = True)


quantsales=np.average(np.stack((
            sales.iloc[:,-364:].quantile(np.array([0.005,0.025,0.165,0.25, 0.5, 0.75, 0.835, 0.975, 0.995]), axis=1).T ,
#             sales.iloc[:,-2*366:].quantile(np.array([0.005,0.025,0.165,0.25, 0.5, 0.75, 0.835, 0.975, 0.995]), axis=1).T ,
            sales.iloc[:,-28:].quantile(np.array([0.005,0.025,0.165,0.25, 0.5, 0.75, 0.835, 0.975, 0.995]), axis=1).T ,
#             sales.iloc[:,-366-365:].quantile(np.array([0.005,0.025,0.165,0.25, 0.5, 0.75, 0.835, 0.975, 0.995]), axis=1).T ,
#             sales.quantile(np.array([0.005,0.025,0.165,0.25, 0.5, 0.75, 0.835, 0.975, 0.995]), axis=1).T
            )),axis=0,weights=[1,1.75])


quantsalesW=[]
for i in range(7):
#     quantsalesW.append(np.expand_dims((sales.iloc[:,np.arange(-28*13+i,0,7)].quantile(np.array(
#         [0.005,0.025,0.165,0.25, 0.5, 0.75, 0.835, 0.975, 0.995]), axis=1).T).values.reshape(quantsales.shape[0]*quantsales.shape[1], order='F'),-1) )
    quantsalesW.append( 
        np.expand_dims(
        np.average(
            np.stack(
        ((sales.iloc[:,np.arange(-28*13+i,0,7)].quantile(np.array(
        [0.005,0.025,0.165,0.25, 0.5, 0.75, 0.835, 0.975, 0.995]), axis=1).T).values.reshape(quantsales.shape[0]*quantsales.shape[1], order='F'),
        (sales.iloc[:,np.arange(-28*3+i,0,7)].quantile(np.array(
        [0.005,0.025,0.165,0.25, 0.5, 0.75, 0.835, 0.975, 0.995]), axis=1).T).values.reshape(quantsales.shape[0]*quantsales.shape[1], order='F')) 
         ,-1   ),axis=-1), -1)
                      )
quantsalesW=np.hstack(quantsalesW)



quantsalesW=np.tile(quantsalesW,4)


medians=np.where(np.array([float(x.split('_')[-2]) for x in df.iloc[:274410,0]] )== 0.5)[0]
notmedian=np.array([x for x in np.arange(274410) if x not in medians])

# Overwrite Level12 
df.iloc[notmedian,1:] = (0.2*df.iloc[notmedian,1:] + 0.7*np.repeat(np.expand_dims(quantsales.reshape(quantsales.shape[0]*quantsales.shape[1], order='F'),-1),28,1)[notmedian,:]
                       + 0.1*quantsalesW[notmedian,:])

df.iloc[medians,1:] = (0.8*df.iloc[medians,1:] + 0.2*np.repeat(np.expand_dims(quantsales.reshape(quantsales.shape[0]*quantsales.shape[1], order='F'),-1),28,1)[medians,:]
                       )

# Statistical computatition to help overwrite Level of 'state_id','item_id' group
quantsalesdf=pd.DataFrame(quantsales)
quantsalesdf['item_id']=sales['item_id'].values
quantsalesdf['state_id']=sales['state_id'].values
quantsalesdfGB=quantsalesdf.groupby(['state_id','item_id'],as_index=False).mean().iloc[:,2:]
#quantsalesdfGB


salesGitemQ=quantsalesdfGB.values.reshape(quantsalesdfGB.shape[0]*quantsalesdfGB.shape[1], order='F')
salesGitemQ=np.repeat(np.expand_dims(salesGitemQ,-1),28,1)
#salesGitemQ.shape



medians=np.where(np.array([float(x.split('_')[-2]) for x in df.iloc[302067:302067+3049*3*9,0]] )== 0.5)[0]
notmedian=np.array([x for x in np.arange(302067,302067+3049*3*9) if x not in medians+302067])
# Overwrite Level's predictions
df.iloc[notmedian,1:] = 0.91*df.iloc[notmedian,1:]  +0.09*salesGitemQ[notmedian-302067,:]


#qals=np.array([float(x.split('_')[-2]) for x in df.id])

# copy preds from first have to second half, predictions for public LB are not right
df.iloc[int(df.shape[0]/2):,1:]=df.iloc[:int(df.shape[0]/2),1:].values


# Create final submission file
df.to_csv(SUBMISSION_DIR+"submission_uncertainty.csv", index = False)


