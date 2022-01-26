#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 10:49:43 2020

@author: malom
"""

import numpy as np
import os
import scipy
#RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
#from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn import metrics
import time

def apply_RF_train_test(x_features_RF, y_train, x_test_feature,y_test):    
    RF_model = RandomForestClassifier(n_estimators = 50, random_state = 42)
    # Train the model on training data
    RF_model.fit(x_features_RF, y_train) #For sklearn no one hot encoding
    #Now predict using the trained RF model. 
    prediction_RF = RF_model.predict(x_test_feature)
    #Inverse le transform to get original label back. 
    #prediction_RF = le.inverse_transform(x_test_feature)
    
    #Now predict using the trained RF model. 
    ts_start = time.time()
    prediction_RF = RF_model.predict(x_test_feature) #This is out X input to RF
    ts_end = time.time()
    ttst = ts_start-ts_end
    #Print overall accuracy
    #from sklearn import metrics
    RF_testing_acc = metrics.accuracy_score(y_test, prediction_RF)
    print ("Accuracy for RF = ", RF_testing_acc)
    
    return RF_testing_acc, prediction_RF, RF_model,ttst



