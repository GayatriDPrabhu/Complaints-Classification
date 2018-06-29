
# coding: utf-8

# In[69]:

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize


# In[70]:

get_ipython().magic('config IPCompleter.greedy=True')


# In[71]:

data = pd.read_csv("Consumer_Complaints.csv");


# In[72]:

data = data[['Product', 'Consumer complaint narrative']];
data.head()


# In[73]:

data = data[pd.notnull(data['Consumer complaint narrative'])];
data.head()


# In[74]:

data['Category_Id'] = data['Product'].factorize()[0];
data = data[['Product', 'Category_Id', 'Consumer complaint narrative']];
data.head()


# In[75]:

data_category_and_id = data[['Product', 'Category_Id']].drop_duplicates().sort_index()
data_category_and_id


# In[76]:

category_to_id = dict(data_category_and_id.values)
category_to_id


# In[77]:

id_to_category = dict(data_category_and_id[['Category_Id', 'Product']].values)
id_to_category


# In[78]:

data.groupby('Product')['Consumer complaint narrative'].count().plot.bar();
plt.show();


# In[79]:

data.shape


# In[80]:

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=0.1, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english');


# In[81]:

features = tfidf.fit_transform(data['Consumer complaint narrative'])


# In[82]:

features_array = features.toarray()


# In[83]:

tfidf.vocabulary_


# In[84]:

ps = PorterStemmer()
data_sample = data[0:10]
data_sample


# In[85]:

new_array = []
for sentence in data_sample['Consumer complaint narrative']:
    new = ""
    for word in sentence.split(" "):
        new += ps.stem(word)+" "
    new_array.append(new)
data_sample['stemmed_disc'] = new_array


# In[86]:

data_sample

