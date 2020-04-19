#!/usr/bin/env python
# coding: utf-8

# In[7]:


import nltk


# In[9]:


from nltk.corpus import brown


# In[10]:


# Corpus- Large collection of text
brown.categories()


# In[11]:


data = brown.sents(categories='fiction')


# In[18]:


print(" ".join(data[1]))


# In[19]:


from nltk.corpus import stopwords

sw = set(stopwords.words('english'))


# In[21]:


print(len(sw))


# In[22]:


def remove_stopwords(words,sw):
    useful_words = []
    useful_words = [w for w in words if w not in sw]
    return useful_words


# In[24]:


setence = "I do not love her very much"
ans = remove_stopwords(setence.split(),sw)
print(ans)


# In[ ]:




