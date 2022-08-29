#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd


# In[5]:


import matplotlib.pyplot as plt


# In[6]:


import numpy as np


# In[7]:


import seaborn as sns


# In[8]:


import scipy
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison


# In[9]:


data = pd.read_csv('C:/Users/Owner/Downloads/hour.csv')


# In[ ]:





# In[10]:


data


# In[11]:


Plot = data.plot.scatter(x='casual', y='cnt')
Plot.set_title("Temp by Humidity")
Plot.set_xlabel("temps")
Plot.set_ylabel("humidity")


# In[12]:


Plot = data.plot.scatter(x='registered', y='cnt')
Plot.set_title("Temp by Humidity")
Plot.set_xlabel("temps")
Plot.set_ylabel("humidity")


# In[13]:


Plot = data.plot.scatter(x='weekday', y='cnt')
Plot.set_title("Temp by Humidity")
Plot.set_xlabel("temps")
Plot.set_ylabel("humidity")


# In[14]:


Plot = data.plot.scatter(x='season', y='cnt')
Plot.set_title("Temp by Humidity")
Plot.set_xlabel("season")
Plot.set_ylabel("cnt")


# In[15]:


plt.plot(data['dteday'], data['cnt'])
plt.xlabel('Date')
plt.ylabel('Count')
plt.title("Counts over Time")


# In[16]:


data.describe()


# In[17]:


data.hist(figsize=(12,22))


# In[18]:


corPearson = data.corr(method="pearson")


# In[19]:


figure = plt.figure(figsize=(15,8))
sns.heatmap(corPearson,annot=True)
plt.show()


# In[20]:


sns.pairplot(data=data, hue='season');
plt.subtitle('Pair Plot of Hour Dataset')


# In[23]:


categories = ['1', '2','3','4']
data1 = data['season'].isin(categories)
data2 = data[data1].copy()


# In[24]:


data3 = data2[['season','cnt']]


# In[25]:


data3.info()


# In[29]:


def recode (series):
    if series == "1": 
        return 0
    if series == "2": 
        return 1
    if series == "3": 
        return 2
    if series == "4": 
        return 3

data3['seasonR'] = data3['season'].apply(recode)


# In[30]:


data4 = data3[['seasonR','cnt']]


# In[35]:


sns.distplot(data4['cnt'])


# In[34]:


data4['cntSQRT'] = np.sqrt(data4['cnt'])


# In[36]:


scipy.stats.bartlett(data4['cntSQRT'], data4['seasonR'])


# In[ ]:




