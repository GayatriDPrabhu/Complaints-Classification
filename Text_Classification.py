
# coding: utf-8

# In[13]:

import pandas as pd
import matplotlib.pyplot as plt


# In[5]:

get_ipython().magic('config IPCompleter.greedy=True')


# In[6]:

data = pd.read_csv("Consumer_Complaints.csv");


# In[7]:

data = data[['Product', 'Consumer complaint narrative']];
data.head()


# In[8]:

data = data[pd.notnull(data['Consumer complaint narrative'])];
data.head()


# In[9]:

data['Category_Id'] = data['Product'].factorize()[0];
data = data[['Product', 'Category_Id', 'Consumer complaint narrative']];
data.head()


# In[10]:

data_category_and_id = data[['Product', 'Category_Id']].drop_duplicates().sort_index()
data_category_and_id


# In[11]:

category_to_id = dict(data_category_and_id.values)
category_to_id


# In[12]:

id_to_category = dict(data_category_and_id[['Category_Id', 'Product']].values)
id_to_category


# In[26]:

data.groupby('Product')['Consumer complaint narrative'].count().plot.bar();
plt.show();

