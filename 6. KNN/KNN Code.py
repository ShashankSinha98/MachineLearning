#!/usr/bin/env python
# coding: utf-8

# In[55]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics


# In[56]:


df = pd.read_csv('mnist_train.csv')
print(df.shape)


# In[57]:


df.head(n=5)


# In[58]:


data = df.values
print(data.shape)
print(type(data))


# In[59]:


X = data[:,1:]
Y = data[:,0]

print(X.shape,Y.shape)


# In[60]:


split = int(.8*X.shape[0])
print("Split:",split)

X_train = X[:split,:]
Y_train = Y[:split]

X_test = X[split:,:]
Y_test = Y[split:]

print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)


# In[61]:


# Visualize Img
def drawImg(sample):
    img = sample.reshape((28,28))
    plt.imshow(img,cmap='gray')
    plt.show()
    
    
drawImg(X_train[3])


# In[ ]:





# In[62]:


def dist(x1,x2):
    return np.sqrt(sum((x1-x2)**2))

def KNN(X,Y,query_point,K=5):
    vals = []
    m = X.shape[0]
    
    for i in range(m):
        d = dist(query_point,X[i])
        vals.append((d,Y[i]))
        
    vals = sorted(vals)
    # Nearest/First K points
    vals = vals[:K]
    
    vals = np.array(vals)
    print("Vals: ",vals)
    new_vals = np.unique(vals[:,1],return_counts=True)
    print(new_vals)
    
    index = new_vals[1].argmax()
    pred = new_vals[0][index]
    
    return pred
    


# In[70]:


pred = KNN(X_train,Y_train,X_test[8])
print(int(pred))


# In[69]:


drawImg(X_test[8])
print(Y_test[8])


# In[66]:


# Accuracy of KNN
Y_Pred = []
Y_Tst = Y_test[:10]
for i in range(10):
    p = KNN(X_train,Y_train,X_test[i])
    Y_Pred.append(p)
    
print("Accuracy: ",np.mean(Y_Pred==Y_Tst))


# In[68]:


print(Y_Pred)
print(Y_Tst)


# In[ ]:




