#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd


# In[6]:


files = [f for f in os.listdir() if '.csv' in f and 'val' in f and 'submission' in f]
len(files)


# In[11]:


sorted(files)


# In[12]:


s = []
for file in files:
    s.append(pd.read_csv(file, header = 0))
    


# In[15]:


s = pd.concat(s)


# In[16]:


s


# In[ ]:





# In[19]:


import zipfile


# In[17]:


s.to_csv('submission.csv', index = False)


# In[22]:


# zipfile.ZipFile('submission.zip', 'w', zipfile.ZIP_DEFLATED).write('submission.csv')


# In[ ]:




