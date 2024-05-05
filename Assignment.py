#!/usr/bin/env python
# coding: utf-8

# # Decision tree algorithms

# 1)CART Algorithm
# 2) ID3 Algorithm
# 3)C4.5 Algorithm
# 

# In[1]:


get_ipython().system('pip install Chefboost')


# In[2]:


from chefboost import Chefboost as chef
import pandas as pd


# In[3]:


df=pd.read_csv("C:/Users/vinod/OneDrive/Desktop/3rd year/lab/dv lab/dataset/diabetes.csv")
df


# # Using ID3 algorithm

# In[4]:


config = {'algorithm': 'ID3'}


mod = chef.fit(df, config = config, target_label = 'Outcome')


# In[5]:


pred= chef.predict(mod, param = [6,148,72,35,0,33.6,0.627,50])
pred


# In[6]:


pred= chef.predict(mod, param = [10,101,76,48,180,32.9,0.171,63])
pred


# # Using CART algorithm

# In[7]:


config = {'algorithm': 'CART'}

model = chef.fit(df, config = config, target_label = 'Outcome')


# In[8]:


pred= chef.predict(model, param = [6,148,72,35,0,33.6,0.627,50])
pred


# In[9]:


pred= chef.predict(model, param = [10,101,76,48,180,32.9,0.171,63])
pred


# # Using C4.5 Algorithm

# In[10]:


config = {'algorithm': 'C4.5'}

mo = chef.fit(df, config = config, target_label = 'Outcome')


# In[11]:


pred= chef.predict(mo, param = [10,101,76,48,180,32.9,0.171,63])
pred


# In[12]:


pred= chef.predict(mo, param = [1,126,60,0,0,30.1,0.349,47,1])
pred


# In[ ]:




