# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 17:45:46 2018

@author: evaljy
"""
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
import pickle
# Read data 
def data2XY(name):
    df = pd.read_csv(name)
    x = df.loc[:, df.columns != 'Empathy'].iloc[:,1:]
    y = df['Empathy']
    return x,y

def baseline_models():
    np.random.seed(1)
    # reading data
    print("begin reading")
    trainX,trainY = data2XY('train.csv')
    validateX,validateY = data2XY('validate.csv')
    
    # decision tree
    acc_tree = []
    model_tree = []
    for i in range(100,1000,100):
        clf = DecisionTreeClassifier(max_depth = i)
        clf.fit(trainX,trainY)
        acc_tree.append(sum(clf.predict(validateX)==validateY)/len(validateX))
        model_tree.append(clf)
    # save model
    print("For decision tree, when the number of trees is", (acc_tree.index(max(acc_tree))+1)*100, ", we reach the highest accuracy ",max(acc_tree))
    # save random forest models
    pickle.dump(model_tree[acc_tree.index(max(acc_tree))], open('dt.db', 'wb'))
    
    
    # svm
    clf = LinearSVC(tol=1e-5)
    clf.fit(trainX,trainY)
    acc_svm = sum(clf.predict(validateX)==validateY)/len(validateX)
    model_svm = clf
    # save model
    print("For SVM, the accuracy is ", acc_svm)
    # save random forest models
    pickle.dump(model_svm, open('svm.db', 'wb'))