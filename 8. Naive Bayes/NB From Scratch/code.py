#!/usr/bin/env python
# coding: utf-8

# In[123]:


import pandas as pd 
import numpy as np 
from collections import defaultdict
import re


# In[124]:


def preprocess_string(str_arg):
    cleaned_str=re.sub('[^a-z\s]+',' ',str_arg,flags=re.IGNORECASE) #every char except alphabets is replacedd
    cleaned_str=re.sub('(\s+)',' ',cleaned_str) #multiple spaces are replaced by single space
    cleaned_str=cleaned_str.lower() #converting the cleaned string to lower case
    
    return cleaned_str # returning the preprocessed string 


# In[125]:


class NaiveBayes:
    
    def __init__(self,unique_classes):
        
        self.classes=unique_classes # Constructor is sinply passed with unique number of classes of the training sett
        
        
    def addToBow(self,example,dict_index):
        
        #print("Ex 1: ",example)
        
        if isinstance(example,np.ndarray):
            example=example[0]
            #print("is instance executed")
        
        #print("Ex 2: ",example)
        #print("dict indx:",dict_index)
        
        for token_word in example.split(): #for every word in preprocessed example
            self.bow_dicts[dict_index][token_word]+=1 #increment in its count
            
            
    def train(self,dataset,labels):
        self.examples=dataset
        self.labels=labels
        self.bow_dicts=np.array([defaultdict(lambda:0) for index in range(self.classes.shape[0])])
        print("Init Bow Dict",self.bow_dicts)
        
        if not isinstance(self.examples,np.ndarray): self.examples=np.array(self.examples)
        if not isinstance(self.labels,np.ndarray): self.labels=np.array(self.labels)
            
        #constructing BoW for each category
        #print(self.labels==0)
        for cat_index,cat in enumerate(self.classes):
            all_cat_examples=self.examples[self.labels==cat]
            
        
            cleaned_examples=[preprocess_string(cat_example) for cat_example in all_cat_examples]
            #print("Cleaned Ex 1: ",cleaned_examples)
            #print("Cleaned Ex 1 type: ",type(cleaned_examples))
        
            cleaned_examples=pd.DataFrame(data=cleaned_examples)
            #print("Cleaned Ex 2: ",cleaned_examples)
            #print("Cleaned Ex 2 type: ",type(cleaned_examples))
            
            #now costruct BoW of this particular category
            np.apply_along_axis(self.addToBow,1,cleaned_examples,cat_index)
            
            prob_classes=np.empty(self.classes.shape[0])
            all_words=[]
            cat_word_counts=np.empty(self.classes.shape[0])
            
            for cat_index,cat in enumerate(self.classes):
                #Calculating prior probability p(c) for each class
                prob_classes[cat_index]=np.sum(self.labels==cat)/float(self.labels.shape[0]) 
            
                #Calculating total counts of all the words of each class 
                count=list(self.bow_dicts[cat_index].values())
                cat_word_counts[cat_index]=np.sum(np.array(count))+1 # |v| is remaining to be added
            
                #get all words of this category                                
                all_words+=self.bow_dicts[cat_index].keys()
                
            #combine all words of every category & make them unique to get vocabulary -V- of entire training set
        
            self.vocab=np.unique(np.array(all_words))
            self.vocab_length=self.vocab.shape[0]
            
            #computing denominator value                                      
            denoms=np.array([cat_word_counts[cat_index]+self.vocab_length+1 for cat_index,cat in enumerate(self.classes)])
            
            
            '''
            Now that we have everything precomputed as well, its better to organize everything in a tuple 
            rather than to have a separate list for every thing.
            
            Every element of self.cats_info has a tuple of values
            Each tuple has a dict at index 0, prior probability at index 1, denominator value at index 2
            '''
        
            self.cats_info=[(self.bow_dicts[cat_index],prob_classes[cat_index],denoms[cat_index]) for cat_index,cat in enumerate(self.classes)]                               
            self.cats_info=np.array(self.cats_info)
            
            
    def getExampleProb(self,test_example):
        likelihood_prob=np.zeros(self.classes.shape[0]) #to store probability w.r.t each clasS
        
        #finding probability w.r.t each class of the given test example
        for cat_index,cat in enumerate(self.classes):
            
            for test_token in test_example.split(): #split the test example and get p of each test word
                
                #get total count of this test token from it's respective training dict to get numerator value                           
                test_token_counts=self.cats_info[cat_index][0].get(test_token,0)+1
                
                #now get likelihood of this test_token word                              
                test_token_prob=test_token_counts/float(self.cats_info[cat_index][2])
                
                #remember why taking log? To prevent underflow!
                likelihood_prob[cat_index]+=np.log(test_token_prob)
                
        # we have likelihood estimate of the given example against every class but we need posterior probility
        post_prob=np.empty(self.classes.shape[0])
        for cat_index,cat in enumerate(self.classes):
            post_prob[cat_index]=likelihood_prob[cat_index]+np.log(self.cats_info[cat_index][1])
            
        return post_prob
    
    
    def test(self,test_set):
        
        predictions=[] #to store prediction of each test example
        for example in test_set: 
                                              
            #preprocess the test example the same way we did for training set exampels                                  
            cleaned_example=preprocess_string(example) 
             
            #simply get the posterior probability of every example                                  
            post_prob=self.getExampleProb(cleaned_example) #get prob of this example for both classes
            
            #simply pick the max value and map against self.classes!
            predictions.append(self.classes[np.argmax(post_prob)])
                
        return np.array(predictions) 
    
    
        
        
    def print_data(self):
        print("Bow Dict",self.bow_dicts)
        print("Outer Bow type",type(self.bow_dicts))
        print("Inner Bow type",type(self.bow_dicts[0]))
        print("Bow Dict Shape",self.bow_dicts.shape)
        print("Bow Dict indx:0 ",self.bow_dicts[0])
        print("Bow Dict indx:1 ",self.bow_dicts[1])


# In[126]:


import numpy as np
#x = ["This was an awesome movie",
#    "Great Movie! I liked it a lot.",
#    "Happy ending! Awesome acting by the hero",
#     "Loved it! Truly great",
#    "bad. not upto mark",
#     "could have been better",
#    "surely a disappointing movie"]

#y = [1,1,1,1,0,0,0]


# In[91]:


#y_labels = np.unique(y)


# In[92]:


#nb = NaiveBayes(y_labels)


# In[93]:


#nb.train(x,y)


# In[88]:


#nb.print_data()


# In[110]:


#a = {"b":2,"c":3}
#d = defaultdict(lambda:"Not Present")
#d["a"]=1
#d["b"]=2
#l1 = list(d.values())
#print(l1)
#l2 = list(d)
#print(l2)
#print(type(d))


# In[115]:


from sklearn.datasets import fetch_20newsgroups


# In[116]:


categories=['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med'] 
newsgroups_train=fetch_20newsgroups(subset='train',categories=categories)


# In[117]:


train_data=newsgroups_train.data #getting all trainign examples
train_labels=newsgroups_train.target #getting training labels


# In[127]:


print ("Total Number of Training Examples: ",len(train_data)) # Outputs -> Total Number of Training Examples:  2257
print ("Total Number of Training Labels: ",len(train_labels)) # Outputs -> #Total Number of Training Labels:  2257


# In[128]:


nb=NaiveBayes(np.unique(train_labels)) #instantiate a NB class object
print ("---------------- Training In Progress --------------------")


# In[129]:


nb.train(train_data,train_labels) #start tarining by calling the train function
print ('----------------- Training Completed ---------------------')


# In[130]:


newsgroups_test=fetch_20newsgroups(subset='test',categories=categories) #loading test data
test_data=newsgroups_test.data #get test set examples
test_labels=newsgroups_test.target #get test set labels

print ("Number of Test Examples: ",len(test_data)) # Output : Number of Test Examples:  1502
print ("Number of Test Labels: ",len(test_labels)) # Output : Number of Test Labels:  1502


# In[131]:


pclasses=nb.test(test_data) #get predcitions for test set

#check how many predcitions actually match original test labels
test_acc=np.sum(pclasses==test_labels)/float(test_labels.shape[0]) 

print ("Test Set Examples: ",test_labels.shape[0]) # Outputs : Test Set Examples:  1502
print ("Test Set Accuracy: ",test_acc*100,"%") # Outputs : Test Set Accuracy:  93.8748335553 %


# In[ ]:




