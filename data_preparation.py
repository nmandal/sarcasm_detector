## Imports
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


## Settings
RANDOM_SEED = 1


## Data Preparation
class DataPrep():
    
    def __init__(self):
        PATH_TO_DATA = 'sarcasm/train-balanced-sarcasm.csv'
        train_data = pd.read_csv(PATH_TO_DATA)

        # Some comments are missing, so we drop corresponding rows
        train_data.dropna(subset=['comment'], inplace=True)

        # Split all data into training and testing
        self.X_train, self.X_test, self.y_train, self.y_test =                 train_test_split(train_data['comment'], train_data['label'], random_state=RANDOM_SEED)

        # Split training data into training and validation
        self.X_train, self.X_valid, self.y_train, self.y_valid =                 train_test_split(self.X_train, self.y_train, random_state=RANDOM_SEED)
        
        # Label encode the target variable
        le = preprocessing.LabelEncoder()
        self.y_train = le.fit_transform(self.y_train)
        self.y_valid = le.fit_transform(self.y_valid)
        self.y_test = le.fit_transform(self.y_test)
        
        self.X_train.reshape(-1,1)
        self.y_train.reshape(-1,1)
    
        self.X_valid.reshape(-1,1)
        self.y_valid.reshape(-1,1)
    
        self.X_test.reshape(-1,1)
        self.y_test.reshape(-1,1)
 