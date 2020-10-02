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

pd.options.display.max_columns = 50
pd.options.display.max_rows = 50
        
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


sub = pd.read_csv('./proc_data/partial_submission.csv')
factors = pd.read_csv('./proc_data/factor.csv')
df_ids = pd.read_csv('./raw_data/sales_train_evaluation.csv', usecols=['id','dept_id','store_id']).drop_duplicates()


# In[3]:


sub = sub.drop_duplicates()


# In[4]:


sub


# In[5]:


factors = df_ids.merge(factors, on=['dept_id','store_id']).drop(['dept_id','store_id'], axis=1)
sub = sub.merge(factors, on='id', how='inner')


# In[6]:


sub


# In[7]:


cols_f = ['F{}'.format(c) for c in range(1, 29)]
for c in cols_f:
    sub[c] = sub[c]*sub['factor']
    
sub = sub[['id']+cols_f]


# In[8]:


sub


# In[9]:


sub2 = sub.copy()
sub2["id"] = sub2["id"].str.replace("evaluation","validation")
sub = pd.concat([sub2, sub], axis=0, sort=False)
sub.columns = ['id'] + ['F' + str(c) for c in np.arange(1,29,1)]


# In[10]:


sub.to_csv('./output_data/submission.csv', index=False)


# In[11]:


sub

