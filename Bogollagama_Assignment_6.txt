# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 12:27:20 2017

@author: abogollagama
"""
# Predict 422 - Assignment 6
# import base packages into the namespace for this program
import numpy as np
import pandas as pd

# visualization utilities

import matplotlib.pyplot as plt
import time
import seaborn as sns; sns.set()


# --------------------------------------------------------
# Define Classifier 
from sklearn import metrics 

# specify the set of classifiers being evaluated
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix 


clf = RandomForestClassifier(max_features = "sqrt", bootstrap= True, n_estimators = 10)

#--------------------------------------------------------
# Import Data 
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
mnist  # show structure of datasets Bunch object from Scikit Learn

# define arrays from the complete data set
mnist_X, mnist_y = mnist['data'], mnist['target']

print('\n Structure of explanatory variable array:', mnist_X.shape)
print('\n Structure of response array:', mnist_y.shape)

train_X = mnist_X[0:60000,]
train_y = mnist_y[0:60000,].astype(int)

test_X = mnist_X[60000:70000,]
test_y = mnist_y[60000:70000,].astype(int)

print('\nShape of train_X:', train_X.shape)
print('\nShape of train_y:', train_y.shape)
print('\nShape of test_X:', test_X.shape)
print('\nShape of test_y:', test_y.shape)
#--------------------------------------------------
#Evaluate Model 
start_time = time.clock()


clf.fit(train_X, train_y)

pred_y = clf.predict(test_X)

end_time = time.clock()

run_time = end_time - start_time 

print("Run-Time", run_time)

print (pred_y)

print(metrics.classification_report(pred_y, test_y))

mat = confusion_matrix(test_y, pred_y)
sns.heatmap(mat.T, square = True, annot = True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label');

