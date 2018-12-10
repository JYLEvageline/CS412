# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 17:58:01 2018

@author: evaljy
"""
import pickle
from read import data_process
from train import my_model,data2XY
from baseline import baseline_models
import numpy as np
import matplotlib.pyplot as plt

def test(X,Y,model):
    return sum(model.predict(X)==Y)/len(Y)


if __name__ == "__main__":
    
    data_process('responses.csv')
    # train and save my model, includes feature selection and hyperparameter selection
    my_model()
    # train and save baseline, includes hyperparameter selection
    baseline_models()
    
    # read random forest model
    with open('rf.db', 'rb') as f:
        rf = pickle.load(f)
    
    # read feature model    
    with open('sfm.db', 'rb') as f:
        sfm = pickle.load(f)
        
    # read baseline
    # read decision tree
    with open('dt.db', 'rb') as f:
        dt = pickle.load(f)
    # read SVM
    with open('svm.db', 'rb') as f:
        svm = pickle.load(f)
    
    trainX,trainY = data2XY('train.csv')
    validateX,validateY = data2XY('validate.csv')
    testX,testY = data2XY('test.csv')
    
    # begin test
    acc_dt = [test(validateX,validateY,dt),test(testX,testY,dt)]
    acc_svm = [test(validateX,validateY,svm),test(testX,testY,svm)]
    acc_rf = [test(sfm.transform(validateX),validateY,rf),test(sfm.transform(testX),testY,rf)]
    
    names = ['Decision Tree','SVM','Random Forest']
    subjects = ['Test Accuracy', 'Validation Accuracy']
    scores = [acc_dt,acc_svm,acc_rf]
    bar_width = 0.25
    
    index = np.arange(len(scores[0]))
     
    for i in range(len(names)):
        plt.bar(index+i*bar_width, scores[i], bar_width,  label=names[i])
    plt.legend(loc=1, fontsize='x-small')
    plt.xlim([-0.2,2])
    plt.title('Accuracy comparison')
    plt.xticks(index + bar_width, subjects,rotation=0)
    plt.xlabel('Test/Validation Set')
    plt.ylabel('Accuracy')
    plt.show()
