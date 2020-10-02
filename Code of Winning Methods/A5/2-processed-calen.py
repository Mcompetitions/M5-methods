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


calendar = pd.read_csv('./raw_data/calendar.csv')


# In[3]:


calendar.head()


# In[4]:


dict_day_to_group = {
    1:1, 2:1, 3:1, 4:1, 5:1,
    6:2, 7:2, 8:2, 9:2, 10:2,
    11:3, 12:3, 13:3, 14:3, 15:3,
    16:4, 17:4, 18:4, 19:4, 20:4,
    21:5, 22:5, 23:5, 24:5, 25:5,
    26:6, 27:6, 28:6, 29:6, 30:6, 31:6  
}

calendar['group_day'] = calendar['date'].str[-2:].apply(int).map(dict_day_to_group)
calendar = calendar.drop(['date','weekday'], axis=1)
calendar = calendar.rename(columns={'d':'day'})


# In[5]:


calendar = pd.concat([calendar, pd.get_dummies(calendar[['event_name_1','event_name_2']])], axis=1)
calendar = calendar.drop(['event_name_1','event_name_2','event_type_1','event_type_2'], axis=1)


# In[6]:


calendar


# In[7]:


for col in [c for c in calendar.columns.tolist() if 'event_name' in c]:
    days_event = np.where(calendar[col] == 1)[0].tolist()
    calendar[col] = calendar['day']

    dict_days_event = {}
    for d in days_event:
        for i in range(0, 30):
            dict_days_event['d_'+str(d-i)] = 30-i

    calendar[col] = calendar[col].map(dict_days_event).fillna(0)


# In[8]:


calendar['event_sum'] = calendar[[c for c in calendar.columns.tolist() if 'event_name' in c]].sum(axis=1).tolist()


# In[9]:


calendar.head(10)


# In[10]:


calendar.to_csv('./proc_data/processed_calendar.csv', index=False)

