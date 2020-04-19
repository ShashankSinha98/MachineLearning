#!/usr/bin/env python
# coding: utf-8

# In[87]:


from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time


# In[105]:


boston = load_boston()
X = boston.data #(506,13)
y = boston.target #(506,)


# In[106]:


#print(boston.feature_names)
#print(boston.DESCR)

df = pd.DataFrame(X)
df.columns = boston.feature_names
df.head()
#df.describe()


# In[107]:


# Normalising Data
u = np.mean(X, axis = 0)
std = np.std(X, axis = 0)

X = (X - u)/std

df = pd.DataFrame(X)
df.columns = boston.feature_names
df.head()


# In[108]:


### Linear Regression
# theta - (13,)
# X = (506,13)
# m - 506, n-13
# Hypothesis Fn - x is a vector, o/p- value
ones = np.ones((X.shape[0],1))
X = np.hstack((ones,X))
def hypothesis(X, theta):
    return np.dot(X, theta)


# In[109]:


# Error Fn- o/p = value
def error(X,y,theta):
    e = 0.0
    m = X.shape[0]
    y_ = hypothesis(X,theta)
    e = np.sum((y-y_)**2)    
    return e/m


# In[110]:


# Gradient Fn- o/p = (n,)
def gradient(X,y,theta):
    y_ = hypothesis(X,theta)
    grad = np.dot(X.T,(y_-y))
    m = X.shape[0]  
    return grad/m


# In[111]:


# Gradient Descent- o/p = (n,)
def gradient_descent(X,y,learning_rate=0.1, max_epochs=300):
    
    n = X.shape[1]
    theta = np.zeros((n,))
    error_list = []
    
    for i in range(max_epochs):
        e = error(X,y,theta)
        error_list.append(e)
        
        grad = gradient(X,y,theta)
        theta = theta - learning_rate * grad
        return theta, error_list


# In[112]:


start = time.time()
theta, error_list  = gradient_descent(X,y)
end = time.time()
print("Time taken: ",end-start)

plt.plot(error_list)
plt.show()


# In[113]:


def r2_score(Y, Y_): 
    num = np.sum((Y - Y_)**2)
    den = np.sum((Y - Y.mean())**2)  
    score = (1-num/den)
    return score*100


# In[114]:


y_ = []
m = X.shape[0]
for i in range(m):
    pred = hypothesis(X[i], theta)
    y_.append(pred)
    
y_ = np.array(y_)

score = r2_score(y,y_)
print(score)


# In[ ]:





# In[ ]:




