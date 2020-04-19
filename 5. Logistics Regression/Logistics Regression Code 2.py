#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


plt.style.use('seaborn')


# In[3]:


mean_01 = np.array([1, 0.5])
cov_01 = np.array([[1, 0.1],[0.1, 1.2]])

mean_02 = np.array([4, 5])
cov_02 = np.array([[1.21, 0.1],[0.1, 1.3]])

# Normal Distribution
dist_01 = np.random.multivariate_normal(mean_01,cov_01,500) 
dist_02 = np.random.multivariate_normal(mean_02,cov_02,500)
#print(dist_01[:5,:]) #print(dist_02[:5,:])
print(dist_02.shape)


# In[4]:


# Data Visualize
plt.scatter(dist_01[:,0],dist_01[:,1], label='class 0')
plt.scatter(dist_02[:,0],dist_02[:,1], label='class 1',color='r',marker='^')
plt.xlim(-5,10)
plt.ylim(-5,10)
plt.xlabel('X1')
plt.xlabel('X2')
plt.legend()
plt.show()


# In[5]:


# Create training and testing set 
data = np.zeros((1000,3)) 
data[:500,:2] = dist_01 
data[500:,:2] = dist_02 

data[500:,-1] = 1.0

print(data)
np.random.shuffle(data)
split = int(.8*data.shape[0]) 
print("Split: ",split) 

###################################### 
X_train = data[:split,:-1] 
X_test = data[split:,:-1]
Y_train = data[:split,-1] 
Y_test = data[split:,-1]
print(X_train.shape, Y_train.shape) 
print(X_test.shape, Y_test.shape)


# In[6]:


def sigmoid(z):
    return (1.0)/(1+np.exp(-z))


# In[7]:


def predict(X,weights):
    """X-> m*(n+1) matrix, W-> (n+1,) vector
       y = x0.theta0 + x1.theta1 + x2.theta2
       Pred => h(y) = sigmoid(y)
       # O/P - (m,) for I/P X - (m,n+1) and W - (n+1,)
       else for single example, it will be float value.
    """
    z = np.dot(X,weights)
    prediction = sigmoid(z)
    return prediction


# In[8]:


def loss(X,Y,weights):
    """Binary Cross Entropy
       Loss Fn = Sum for all example m {( Yi.log(Yi_) + (1-Yi).log(1-Yi_) )}
       Yi = actual value of example Xi
       Yi_ = predicted value of example Xi
       we take mean of loss of examples by dividing total loss of all examples by total no of examples
       np.mean() - gives mean directly
       # O/P - (m,) for I/P X - (m,n+1), Y- (m,) and W - (n+1,)
       else for single example, it will be float value.
    """
    Y_ = predict(X,weights)
    cost = np.mean(-Y*np.log(Y_) - (1-Y)*np.log(1-Y_))
    return cost


# In[9]:


def update(X,Y,weights,learning_rate):
    """Perform weight update for 1 epoch
       # O/P - (n+1,) for I/P X - (m,n+1), Y- (m,), W - (n+1,) and learning_rate - constant float
       even for single example, it will be same (n+1,)
       Weight Update- d(Loss Fn)/dWj = (Y - Y_).Xj for jth weight
       we are taking -ve of update rule, i.e, Gradient Descent 
    """
    Y_ = predict(X,weights)
    dw = np.dot(X.T,Y_ - Y) # (n+1,), but it contains sum of dw of all examples, so we need to take mean 
                            # of it by dividing it by total no of examples in i/p i.e, m 
    
    m = X.shape[0]
    weights = weights - learning_rate*dw/(float(m))
    return weights


# In[10]:


def train(X,Y,learning_rate=0.8,maxEpochs=100):
    
    # Modify the input to handle the bias term
    ones = np.ones((X.shape[0],1))
    X = np.hstack((ones,X))
    
    # Init Weights
    weights = np.zeros(X.shape[1]) # n+1 entries
    
    for epoch in range(maxEpochs):
        # Iterate over all epochs and make update
        weights = update(X,Y,weights,learning_rate)
        
        if epoch%10==0:
            l = loss(X,Y,weights)
            print("Epoch %d Loss %.4f"%(epoch,l))
            
            
    return weights 
    
    


# In[11]:


weights = train(X_train,Y_train, maxEpochs=1000)


# In[12]:


print(weights)


# In[13]:


x1 = np.linspace(-4,10,20)
x2 = -(weights[0] + weights[1]*x1)/weights[2]


# In[14]:


# Data Visualize
plt.scatter(dist_01[:,0],dist_01[:,1], label='class 0')
plt.scatter(dist_02[:,0],dist_02[:,1], label='class 1',color='r',marker='^')
plt.xlim(-5,10)
plt.ylim(-5,10)
plt.plot(x1,x2,color='red')
plt.xlabel('X1')
plt.xlabel('X2')
plt.legend()
plt.show()


# In[15]:


def getPredictions(X_Test,weights,labels=True):
       
       if X_Test.shape[1] != weights.shape[0]:
           ones = np.ones((X_Test.shape[0],1))
           X_Test = np.hstack((ones,X_Test))
           
       probs = predict(X_Test,weights)
       
       if not labels:
           return probs
       else:
           labels = np.zeros(probs.shape)
           labels[probs>=0.5] = 1
           return labels


# In[16]:


# Find accuracy
Y_ = getPredictions(X_test,weights,labels=True)
training_accuracy = np.sum(Y_==Y_test)/Y_test.shape[0]
print(training_accuracy)


# In[ ]:




