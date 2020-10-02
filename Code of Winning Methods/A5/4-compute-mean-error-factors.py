#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
pd.options.display.max_columns = 100
pd.options.display.max_rows = 100

from matplotlib import pyplot as plt
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


cols_days = ['d_{}'.format(c) for c in range(1914,1942)]


# In[3]:


df_eval = pd.read_csv('./raw_data/sales_train_evaluation.csv', usecols=['id','dept_id','store_id']+cols_days)
df_model = pd.read_csv('./proc_data/partial_submission.csv')


# In[4]:


df_comparison = df_eval.merge(df_model, on='id', how='inner')


# In[5]:


df_comparison.groupby(['dept_id','store_id']).mean()


# In[6]:


df_comp_mean = df_comparison.groupby(['dept_id','store_id']).mean()


# In[7]:


for i in range(0, len(df_comp_mean)):

    x = df_comp_mean.iloc[i,:28].index
    y_true = df_comp_mean.iloc[i,:28].values
    y_pred = df_comp_mean.iloc[i,28:].values
    indice = df_comp_mean.index[i]

    fig, ax = plt.subplots(figsize = (10,3), dpi=150)

    fig.suptitle('Dept: {dept}     Store: {store}'.format(dept=indice[0], store=indice[1]))
    ax.plot(x, y_true, label='True')
    ax.plot(x, y_pred, label='Pred')
    ax.tick_params(axis='x', rotation=45)

    ax.legend()

    plt.show()


# In[8]:


depts = []
stores = []
factors = []

for i in range(0, len(df_comp_mean)):

    x = df_comp_mean.iloc[i,:28].index
    y_true = df_comp_mean.iloc[i,:28].values
    y_pred = df_comp_mean.iloc[i,28:].values
    factor = y_true.mean()/y_pred.mean()
    indice = df_comp_mean.index[i]

    depts.append(indice[0])
    stores.append(indice[1])
    factors.append(factor)
    
    fig, ax = plt.subplots(figsize = (10,3), dpi=150)

    fig.suptitle('Dept: {dept}     Store: {store}     Factor: {factor}'.format(
        dept=indice[0], store=indice[1], factor=factor))
    ax.plot(x, y_true, label='True')
    ax.plot(x, y_pred*factor, label='Pred')
    ax.tick_params(axis='x', rotation=45)

    ax.legend()

    plt.show()
    
df_factors = pd.DataFrame(data={'dept_id':depts, 'store_id':stores, 'factor':factors})


# In[9]:


df_factors


# In[10]:


df_factors.to_csv('./proc_data/factor.csv', index=False)

