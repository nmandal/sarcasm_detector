## Imports
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


## Settings
RANDOM_SEED = 1


## Data Preparation
class FeatureEngineer():
    
    def __init__(self, data):
       ## Count Vectors as features
        # create a count vectorizer object
        count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
        count_vect.fit(data.X_train)

        # transform the training and validation data using count vectorizer object
        self.X_train_count =  count_vect.transform(data.X_train)
        self.X_valid_count =  count_vect.transform(data.X_valid)
        self.X_test_count  =  count_vect.transform(data.X_test)
    
    
      ## TF-IDF Vectors as features
        # word level tf-idf
        tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
        tfidf_vect.fit(data.X_train)
    
        self.X_train_tfidf =  tfidf_vect.transform(data.X_train)
        self.X_valid_tfidf =  tfidf_vect.transform(data.X_valid)
        self.X_test_tfidf  =  tfidf_vect.transform(data.X_test)

        # ngram level tf-idf 
        tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
        tfidf_vect_ngram.fit(data.X_train)
    
        self.X_train_tfidf_ngram =  tfidf_vect_ngram.transform(data.X_train)
        self.X_valid_tfidf_ngram =  tfidf_vect_ngram.transform(data.X_valid)
        self.X_test_tfidf_ngram =  tfidf_vect_ngram.transform(data.X_test)

        # characters level tf-idf
        tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
        tfidf_vect_ngram_chars.fit(data.X_train)
    
        self.X_train_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(data.X_train) 
        self.X_valid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(data.X_valid)
        self.X_test_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(data.X_test)
