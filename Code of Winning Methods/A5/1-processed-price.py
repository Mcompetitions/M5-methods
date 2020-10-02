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


# In[2]:


prices = pd.read_csv('./raw_data/sell_prices.csv')


# In[3]:


prices = prices.sort_values(by=['item_id','store_id','wm_yr_wk'])


# In[4]:


prices.sample(10)


# In[5]:


for win in [1, 2, 4, 8]:
    prices['rm_diff_price_{}'.format(win)] = prices[["store_id","item_id","sell_price"]].groupby(
        ["store_id","item_id"])["sell_price"].transform(lambda x : x.rolling(win).mean())
    prices['rm_diff_price_{}'.format(win)] = ((prices['sell_price'] - prices['rm_diff_price_{}'.format(win)])/prices['sell_price']).round(3)


# In[ ]:


prices.iloc[80:100]


# In[7]:


prices.to_csv('./proc_data/prices_processed.csv', index=False)

