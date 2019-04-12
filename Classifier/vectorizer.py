
# coding: utf-8

# In[1]:


from sklearn.feature_extraction.text import HashingVectorizer
import re
import os
import pickle


# In[2]:


stop = pickle.load(open('stopwords.pkl', 'rb'))


# In[3]:


def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) +        ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

vect = HashingVectorizer(decode_error='ignore',
                         n_features=2**21,
                         preprocessor=None,
                         tokenizer=tokenizer)

