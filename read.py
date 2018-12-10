# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 11:32:51 2018

@author: evaljy
"""

import numpy as np
import pandas as pd
from collections import defaultdict

def data_process(name):
    # data reading
    data = pd.read_csv(name)
    print("The size of dataset before data cleaning is:",data.shape)
    print("Are there any NaN values before cleaning?")
    print(data.isnull().any().any())
    
    # data cleaning
    df = data.dropna()
    df.dropna()
    df.isnull().any().any()
    print("The size of dataset after data cleaning is:",df.shape)
    print("Are there any NaN values before cleaning?")
    print(df.isnull().any().any())
    
    # data with str
    v2i = defaultdict(dict)
    i2v = defaultdict(dict)
    for name in df.columns:
        if df[name].dtype != np.number and df[name].dtype != np.int64:
            value2idx = {}
            idx2value = {}
            for idx, value in enumerate(set(df[name])):
                value2idx[value] = idx+1
                idx2value[idx+1] = value
            # save the dictionary for further use
            v2i[name] = value2idx
            i2v[name] = idx2value
            # begin mapping
            df[name] = df[name].map(value2idx)
            
    
    # empathy change to very	empathetic/not very
    name = 'Empathy'
    value2idx = {1:2,2:2,3:2,4:1,5:1}
    idx2value = {1:[4,5],2:[1,2,3]}
    # save the dictionary for further use
    v2i[name] = value2idx
    i2v[name] = idx2value
    # begin mapping
    df[name] = df[name].map(value2idx)
                   
    # choose set
    np.random.seed(1)
    train, test, validate = np.split(df.sample(frac=1), [int(.7*len(df)), int(.9*len(df))])
    print("The size of training dataset is:",train.shape)
    print("The size of test dataset is:",test.shape)
    print("The size of validation dataset is:",validate.shape)
    train.to_csv("train.csv")
    test.to_csv("test.csv")
    validate.to_csv("validate.csv")
