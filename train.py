# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 14:43:50 2018

@author: evaljy
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectFromModel
import pickle
import numpy as np
import matplotlib.pyplot as plt

'''
X, y = make_classification(n_samples=1000, n_features=4,
                            n_informative=2, n_redundant=0,
                            random_state=0, shuffle=False)

clf = RandomForestClassifier(n_estimators=100, max_depth=2,
                              random_state=0)
clf.fit(X, y)
set(clf.predict(trainX))
'''
# Read data 
def data2XY(name):
    df = pd.read_csv(name)
    x = df.loc[:, df.columns != 'Empathy'].iloc[:,1:]
    y = df['Empathy']
    return x,y

# feature selection
def fs(trainX,idx):
    return trainX[idx]
    
def trainXY(X,Y,n):
    rf = RandomForestClassifier(n_estimators = n, random_state = 42)
    rf.fit(X, Y)
    return rf
    
def feature_selection(trainX,trainY):
    rf = trainXY(trainX,trainY,1000)
    
    # calculate the importance of each feature
    labels = list(trainX.columns)
    importances = rf.feature_importances_
    indices=np.argsort(importances)[::-1]
    
    #for f in range(trainX.shape[1]):
    #    print ("%2d) %-*s %f" % (f+1,30,labels[f],importances[indices[f]]) )
    
    # plot 
    plt.title('Feature Importance-RandomForest')
    plt.bar(range(10),importances[indices[0:10]],color='lightblue',align='center')
    plt.xticks(range(10),[labels[i] for i in indices[0:10]],rotation=90)
    plt.xlim([-1,10])
    plt.tight_layout()
    plt.show()
    
    sfm = SelectFromModel(rf, threshold=0.008)
    sfm.fit(trainX, trainY)
    return sfm

        
def validate(trainXI,trainY,validateXI,validateY):
    accuracy = []
    models = []
    for i in range(1,11):
        print("Number of trees are", i*100)
        rf = trainXY(trainXI,trainY,i*100)
        y = rf.predict(validateXI)
        models.append(rf)
        accuracy.append(sum(y==validateY)/len(y))
    return models,accuracy
     

def my_model():
    np.random.seed(1)
    # reading data
    print("begin reading")
    trainX,trainY = data2XY('train.csv')
    validateX,validateY = data2XY('validate.csv')
    
    # feature selection
    print("begin feature selection")
    sfm = feature_selection(trainX,trainY)
    # save model on feature selection
    pickle.dump(sfm, open('sfm.db', 'wb'))
    
    # select parameters
    print("begin hyperparameter selection")
    trainXI = sfm.transform(trainX)
    validateXI = sfm.transform(validateX)
    models,accuracy = validate(trainXI,trainY,validateXI,validateY)
    plt.title('The influence of hyperparameter on validation set')
    plt.bar(range(len(accuracy)),accuracy,color='lightblue',align='center')
    plt.xticks(range(len(accuracy)),range(100,1100,100),rotation=90)
    plt.xlim([-1,10])
    plt.xlabel("number of trees")
    plt.ylabel("accuracy")
    plt.show()
    # save model
    print("For random forest, when the number of trees is", (accuracy.index(max(accuracy))+1)*100, "we reach the highest accuracy ",max(accuracy))
    # save random forest models
    pickle.dump(models[accuracy.index(max(accuracy))], open('rf.db', 'wb'))


