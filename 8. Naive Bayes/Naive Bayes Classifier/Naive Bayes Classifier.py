#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split


# In[12]:


df = pd.read_csv('mushrooms.csv')
df.head(n=5)


# In[13]:


print(df.columns)


# ### Encode the Categorical data into Numerical Data

# In[15]:


le = LabelEncoder()
#Applies transformation on each column
ds = df.apply(le.fit_transform)


# In[17]:


ds.head(n=5)


# In[19]:


data = ds.values
print(data.shape)
print(type(data))
print(data[:5,:])

data_x = data[:,1:]
data_y = data[:,0]


# ### Break the data into Train and Test Set  

# x_train,x_test,y_train,y_test = train_test_split(data_x,data_y,test_size=0.2)
# print(x_train.shape, y_train.shape)
# print(x_test.shape,y_test.shape)

# In[22]:


np.unique(y_train) # only 2 classes of mushroom


# ### Building Classifier

# In[24]:


def prior_prob(y_train, label):
    
    total_examples = y_train.shape[0]
    class_examples = np.sum(y_train==label)
    
    return class_examples/float(total_examples)


# In[29]:


#a = np.array([0,1,1,0,1,1,0,1,0,1])
#b = np.array([10,20,30,40,50,60,70,80,90,100])
#print(a==1)
#c = b[a==1]
#print(c)


# In[35]:


def cond_prob(x_train,y_train,feature_col,feature_val,label):
    
    x_filtered = x_train[y_train==label] 
    num = np.sum(x_filtered[:,feature_col]==feature_val)
    den = np.sum(y_train==label)
    
    return num/float(den)


# ### Compute Posterior Probability for each test example and make predictions

# In[36]:


def predict(X_train,Y_train,x_test):
    
    classes = np.unique(Y_train)
    features = X_train.shape[1]
    post = []
    for label in classes:
        
        likelihood = 1.0
        for f in range(features):
            cond = cond_prob(X_train,Y_train,f,x_test[f],label)
            likelihood *= cond
            
        prior = prior_prob(Y_train,label)
        posterior = likelihood * prior
        post.append(posterior)
        
    return np.argmax(post)


# In[37]:


pred = predict(x_train,y_train,x_test[0])
print(pred)
print(y_test[0])


# In[38]:


def accuracy(X_train,Y_train,X_test,Y_test):
    
    pred = []
    n = X_test.shape[0]
    for i in range(n):
        pred.append(predict(X_train,Y_train,X_test[i]))
        
    accuracy = np.sum(pred==Y_test)/float(n)
    return accuracy


# In[ ]:




