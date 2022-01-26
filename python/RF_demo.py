#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 15:55:58 2021

@author: malom
"""

from __future__ import print_function
import os,time,cv2, sys, math
import tensorflow as tf
import numpy as np
import time, datetime
import argparse
import random
import os,sys,json
import subprocess
from os.path import join as join_path

from random import shuffle
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

import matplotlib
import pdb
import seaborn as sns
from json import JSONEncoder

import ML
matplotlib.use('Agg')
import matplotlib.pyplot as plt

abspath = os.path.dirname(os.path.abspath(__file__))

testing_log_saving_path = '/test_saving_path/'

X_train = []  # N X M (N-number of cases, M number of attributes)
y_train = []  # N X c ( Encoded value of c- number of classes )
X_test = [] 
y_test = []
################################ DL feature and RANDOM FOREST ################################
RF_testing_acc, prediction_RF, RF_model, total_testing_time_rf = ML.apply_RF_train_test(X_train, y_train, X_test, y_test)
print ("Accuracy for RF = ", RF_testing_acc)
pred_saving_path_rf = os.path.join(testing_log_saving_path,'y_pred_rf.npy')
np.save(pred_saving_path_rf,prediction_RF)
#Confusion Matrix - verify accuracy of each class
cm = confusion_matrix(y_test, prediction_RF)
#print(cm)
sns.heatmap(cm, annot=True)
#Check results on a  select samply ........ With RANDOM FOREST>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# n=9 #Select the index of image to be loaded for testing
# input_case = x_test[n]
# #plt.imshow(img)
# input_case = np.expand_dims(input_case, axis=0) #Expand dims so the input is (num images, x, y, c)
# # input_case_features=feature_extractor.predict(input_case)
# prediction_RF = RF_model.predict(input_case_features)[0] 
# #prediction_RF = le.inverse_transform([prediction_RF])  #Reverse the label encoder to original name
# print("The prediction for this case is: ", prediction_RF)
# print("The actual label for this case is: ", y_test[n])

### saving the testing logs <<<<<<<<<<<<<<<<       >>>>>>>>>>>>>>>>>>>>>>>
testing_log = {}
testing_log["Testing accuracy for RF on DL feature: "] = RF_testing_acc
testing_log["Testing time for RF : "] = total_testing_time_rf

json_file = os.path.join(testing_log_saving_path,'testing_log_class13_final.json')
with open(json_file, 'w') as file_path:
    json.dump(testing_log, file_path, indent=4, sort_keys=True)
