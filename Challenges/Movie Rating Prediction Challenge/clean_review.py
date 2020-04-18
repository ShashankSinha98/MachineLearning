#!/usr/bin/env python
# coding: utf-8

# ### NLTK Pipeline
# #### Create a NLTK Pipeline to 'clean' review data
# 
# <ul><li>Load Input File and Read Reviews</li>
# <li>Tokenize</li>
# <li>Remove Stopwords</li>
# <li>Perform Stemming</li>
# <li>Write cleaned data to Output File</li></ul>

# In[1]:


sample_text="""I loved this movie since I was 7 and I saw it on the opening day. It was so touching and beautiful. I strongly recommend seeing for all. It's a movie to watch with your family by far.<br /><br />My MPAA rating: PG-13 for thematic elements, prolonged scenes of disastor, nudity/sexuality and some language."""


# In[33]:


from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import pandas as pd
import sys
import os


# In[34]:


# Init Objects
tokenizer = RegexpTokenizer('[a-zA-z]+')
en_stopwords = set(stopwords.words('english'))
ps = PorterStemmer()


# In[21]:


def getCleanReview(review):
    
    review = review.lower()
    review = review.replace("<br /><br />"," ")
    
    # Tokenize
    tokens = tokenizer.tokenize(review)
    new_tokens = [token for token in tokens if token not in en_stopwords]
    stemmed_tokens = [ps.stem(token) for token in new_tokens]
    
    cleaned_review = ' '.join(stemmed_tokens)
    
    return cleaned_review


# In[22]:


#print(getCleanReview(sample_text))


# In[25]:


def getStemmedDocument(inputFile,outputFile):
    
    out = open(outputFile,'w',encoding='utf8')
    
    with open(inputFile,encoding='utf8') as f:
        reviews = f.readlines()
        
    for review in reviews:
        cleaned_review = getCleanReview(review)
        print((cleaned_review),file=out)
        
    out.close()


# In[39]:


#inputFile = sys.argv[1]
#outputFile = sys.argv[2]
#getStemmedDocument(inputFile,outputFile)
#print(os.listdir())
#getStemmedDocument("unclean.txt","clean.txt")


# In[ ]:




