#!/usr/bin/env python
# coding: utf-8

# In[15]:


from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt


# In[20]:


gnb = GaussianNB()


# In[21]:


X,Y = make_classification(n_samples=200,n_features=2,n_informative=2,n_redundant=0,random_state=4)
plt.scatter(X[:,0],X[:,1],c=Y)
plt.show()


# In[23]:


print(X[0])
print(X.shape) # continuous values features


# In[24]:


gnb.fit(X,Y)


# In[25]:


gnb.score(X,Y) # accuracy


# In[ ]:




