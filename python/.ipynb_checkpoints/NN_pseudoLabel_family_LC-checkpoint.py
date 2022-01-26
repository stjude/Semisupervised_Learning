import pathlib
import numpy as np
from numpy import argmax
import h5py
import time

import pandas as pd
#import dask.dataframe as dd

import tensorflow as tf
from tensorflow import keras

from tensorflow import feature_column
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.utils import to_categorical
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score,f1_score,precision_score,recall_score 
from sklearn.metrics import confusion_matrix

def process_csv(filename):
    df = pd.read_csv(filename, index_col=0)
    y = df.y
    X = df.drop(columns=['y'])
    return (X, y)

def get_beta_in_select_file(beta, filename):
    X_temp, y = process_csv(filename)
    X = beta[ X_temp.index ]
    y = pd.DataFrame(y)
    return X.T, y

def get_x_y_data(filename, num_probe_return=False):
    df = pd.read_csv(filename, header=0, index_col=0)
    probe_data_t = np.array(df)
    num_probes = len(df.columns)-1
    X = probe_data_t[0:, 0:num_probes]
    y = df['y']
    if (num_probe_return == False):
        return X,y
    else:
        return X, y, num_probes
    
def get_oneHotCode_matrix(labels, encoder):
    integer_encoded = encoder.fit_transform(labels)
    num_classes = len(np.unique(labels))
    labels_one_hot = keras.utils.to_categorical(integer_encoded, num_classes)
    return labels_one_hot

def select_probes(data, sd_cutoff):
    probes = data.T
    probes['STD'] = probes.std(axis=1)
    above_threshold = probes[probes["STD"] > sd_cutoff]
    print("shape in select_probes", above_threshold.shape)
    return above_threshold.drop(columns='STD').T

def match_probes(ref_data, data_to_be_matched):
    df = data_to_be_matched.loc[:, ref_data.columns.values]
    return df

def create_model(n_inputs,n_classes):
    model = keras.Sequential()

    model.add(keras.layers.Dense(1000, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
    model.add(keras.layers.Dense(500, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
    model.add(keras.layers.Dense(n_classes, activation="softmax"))
    
    model.compile(optimizer = 'adam', loss= 'categorical_crossentropy', 
                  metrics=['accuracy', 'Precision','Recall', 'mae', 'mse'])
    
    return model

def cross_validate2(data_x, data_y, cv, encoder, cls_weight, sd_cutoff):
    '''
    features are selectiving for the training data set after each CV split
    '''
    cvscores = []
    fold = 1
    for train_index,test_index in cv.split(data_x, data_y):
        #print("X_cv shape = ", data_x.shape, "; y_cv shape = ", data_y.shape)
        x_train_temp = select_probes(data_x.iloc[train_index, :], sd_cutoff)
        x_test_temp = match_probes(x_train_temp, data_x.iloc[test_index, :])
        x_train = np.asarray(x_train_temp).astype(np.float32) ###convert to np.float32 to be used in model.fit
        x_test = np.asarray(x_test_temp).astype(np.float32)
        y_train,y_test = data_y.iloc[train_index, :], data_y.iloc[test_index, :]
        y_train_encode = get_oneHotCode_matrix(y_train.values.ravel(), encoder)
        y_test_encode = get_oneHotCode_matrix(y_test.values.ravel(), encoder)
        
        n_inputs = len(x_train_temp.columns)
        model = create_model(n_inputs, len(np.unique(y_test))) 
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold} ...')
        y_test_intcode = encoder.fit_transform(y_test.values.ravel())
        val_sample_weights = class_weight.compute_sample_weight(cls_weight, y_test_intcode)
        
        hist = model.fit(x_train, y_train_encode,epochs=20, class_weight = cls_weight, batch_size=50, verbose=0)
        scores = model.evaluate(x_test, y_test_encode, sample_weight = val_sample_weights, verbose=0)
        cvscores.append(scores)
        fold = fold+1
    return model, x_train_temp, pd.DataFrame(cvscores, columns= model.metrics_names)

def get_pred_label2(mod, feature, x, y, encoder):
    x_test_temp = match_probes(feature, x)
    Xtest = np.asarray(x_test_temp).astype(np.float32)
    pred_scores = mod.predict(Xtest)
    pred_int_code = np.argmax(pred_scores, axis=1)
    y_pred = encoder.inverse_transform(pred_int_code)
    return y_pred

def accuracy_per_class(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    classes = np.unique(y_true)
    per_class_accuracies = cm.diagonal()/cm.sum(axis=1)
    per_class_acc_wKeys = {}
    for idx, cls in enumerate(classes):
        per_class_acc_wKeys[cls] = per_class_accuracies[idx]
    return per_class_acc_wKeys


def create_per_class_acc_df(class_acc, seed, dataset_name):
    d = pd.DataFrame(columns = ['Seed', 'Dataset'])
    d = d.append(class_acc, ignore_index=True)
    d['Seed'] = seed
    d['Dataset'] = dataset_name
    return d

def evaluate_against_test_set2(mod, feature, x_test, y_test, encoder, cls_weight):
    y_pred = get_pred_label2(mod, feature, x_test, y_test, encoder)
    y_test_encode = encoder.fit_transform(y_test.values.ravel())
    test_sample_weights = class_weight.compute_sample_weight(cls_weight, y_test_encode)
    
    acc=accuracy_score(y_test,y_pred).round(3)
    bal_acc=balanced_accuracy_score(y_test,y_pred, sample_weight = test_sample_weights).round(3)
    rec=recall_score(y_test,y_pred, average='weighted', zero_division=0).round(3)
    prec=precision_score(y_test,y_pred, average='weighted', zero_division=0).round(3)
    f1=f1_score(y_test,y_pred, average='weighted').round(3)
    per_class_acc = accuracy_per_class(y_test, y_pred)
    
    return (acc, bal_acc, rec, prec, f1, per_class_acc)

def append_result(p, df):
    s = pd.Series(p, index=df.columns)
    return df.append(s, ignore_index=True)

def append_testset_results(df, r, seed_name, dataset_name):
    df = append_result([seed_name, dataset_name, 'vs_testset', 'accuracy', r[0]], df)
    df = append_result([seed_name, dataset_name, 'vs_testset', 'balanced_acc', r[1]], df)
    df = append_result([seed_name, dataset_name, 'vs_testset', 'recall_weighted', r[2]], df)
    df = append_result([seed_name, dataset_name, 'vs_testset', 'precision_weighted', r[3]], df)
    df = append_result([seed_name, dataset_name, 'vs_testset', 'f1_weighted', r[4]], df)
    return df

def apply_CV(mod, cv, X, y):
    return cross_validate(mod, X, y, cv=cv, scoring=['balanced_accuracy', 'accuracy', 'recall_weighted'])

def append_cv_results(df, r, seed_name, dataset_name):
    df = append_result([seed_name, dataset_name, 'cross_val', 'loss', r['loss'].mean()], df)
    df = append_result([seed_name, dataset_name, 'cross_val', 'accuracy', r['accuracy'].mean()], df)
    df = append_result([seed_name, dataset_name, 'cross_val', 'precision', r['precision'].mean()], df)
    df = append_result([seed_name, dataset_name, 'cross_val', 'recall', r['recall'].mean()], df)
    df = append_result([seed_name, dataset_name, 'cross_val', 'mae', r['mae'].mean()], df)
    df = append_result([seed_name, dataset_name, 'cross_val', 'mse', r['mse'].mean()], df)

    return df


def evaluate(seed, beta, rkfold, encoder, cls_weight, sd_cutoff=0.3):
    file70LC= '/research/rgs01/home/clusterHome/qtran/Semisupervised_Learning/python/data_family/seed{}_70percLC.csv'.format(seed)
    
    
    file35_GSELC ='/research/rgs01/home/clusterHome/qtran/Semisupervised_Learning/python/data_family/seed{}_35perc_LCgse109379.csv'.format(seed)
    
    file70LC_GSE = '/research/rgs01/home/clusterHome/qtran/Semisupervised_Learning/python/data_family/seed{}_70LC_gse109379.csv'.format(seed)
    file70_GSELC = '/research/rgs01/home/clusterHome/qtran/Semisupervised_Learning/python/data_family/seed{}_70perc_gse109379LC.csv'.format(seed)
    file70LC_GSELC = '/research/rgs01/home/clusterHome/qtran/Semisupervised_Learning/python/data_family/seed{}_70LC_gse109379LC.csv'.format(seed)
    
    fileHoldOut = '/research/rgs01/home/clusterHome/qtran/Semisupervised_Learning/python/data_family/seed{}_holdOutTest.csv'.format(seed)


    #########
    print('\tReading', file70LC)
    X70LC, y70LC = get_beta_in_select_file(beta, file70LC)
    
    ##
    print('\tReading', file35_GSELC)
    X35gseLC, y35gseLC = get_beta_in_select_file(beta, file35_GSELC)
    ##
    print('\tReading', file70LC_GSE)
    X70LCgse, y70LCgse = get_beta_in_select_file(beta, file70LC_GSE)
    print('\tReading', file70_GSELC)
    X70gseLC, y70gseLC = get_beta_in_select_file(beta, file70_GSELC)
    print('\tReading', file70LC_GSELC)
    XLC, yLC = get_beta_in_select_file(beta, file70LC_GSELC)
    
    print('\tReading', fileHoldOut)
    xitest, yitest = get_beta_in_select_file(beta, fileHoldOut)
    ###################
    
    print('4. Evaluate model with 70% LC')    
    nnmodel70LC, features70LC, result_cv_70LC = cross_validate2(X70LC, y70LC, rkfold, encoder, cls_weight, sd_cutoff=sd_cutoff)
    print('\tvalidate against test set')
    result_ts_70LC = evaluate_against_test_set2(nnmodel70LC, features70LC, xitest, yitest, encoder, cls_weight)
    
       
    print('6. Evaluate model with 35% data with High confident GSE109379')
    nnmodel35gseLC, features35gseLC, result_cv_35gseLC = cross_validate2(X35gseLC, y35gseLC, rkfold, encoder, cls_weight, sd_cutoff=sd_cutoff)
    print('\tvalidate against test set')
    result_ts_35gseLC = evaluate_against_test_set2(nnmodel35gseLC, features35gseLC, xitest, yitest, encoder, cls_weight)
       
    ##########    
             
    print('8. Evaluate model with 70% LC + GSE109379')
    nnmodel70LCgse, features70LCgse, result_cv_70LCgse = cross_validate2(X70LCgse, y70LCgse, rkfold, encoder, cls_weight, sd_cutoff=sd_cutoff)
    print('\tvalidate against test set')
    result_ts_70LCgse = evaluate_against_test_set2(nnmodel70LCgse, features70LCgse, xitest, yitest, encoder, cls_weight)

    print('9. Evaluate model with 70% + LC GSE90496 ')    
    nnmodel70gseLC, features70gseLC, result_cv_70gseLC = cross_validate2(X70gseLC, y70gseLC, rkfold, encoder, cls_weight, sd_cutoff=sd_cutoff)
    print('\tvalidate against test set')
    result_ts_70gseLC = evaluate_against_test_set2(nnmodel70gseLC,features70gseLC, xitest, yitest, encoder, cls_weight)
      
    print('10. Evaluate model with LC pseudo-labeled data')    
    nnmodelLC, featuresLC, result_cv_LC = cross_validate2(XLC, yLC, rkfold, encoder, cls_weight, sd_cutoff=sd_cutoff)
    print('\tvalidate against test set')
    result_ts_LC = evaluate_against_test_set2(nnmodelLC, featuresLC, xitest, yitest, encoder, cls_weight)


    
    
    #########
    
    print('6. Store results.')
    df = pd.DataFrame(data=[], columns=['Seed','Dataset','Validation','Metric','Value'])
    df = append_cv_results(df, result_cv_70LC, seed, '70LC')
    df = append_testset_results(df, result_ts_70LC, seed, '70LC')
    
    df = append_cv_results(df, result_cv_35gseLC, seed, '35_gseLC')
    df = append_testset_results(df, result_ts_35gseLC, seed, '35_gseLC')
    
    df = append_cv_results(df, result_cv_70LCgse, seed, '70LC_GSE')
    df = append_testset_results(df, result_ts_70LCgse, seed, '70LC_GSE')
    df = append_cv_results(df, result_cv_70gseLC, seed, '70_gseLC')
    df = append_testset_results(df, result_ts_70gseLC, seed, '70_gseLC')
    df = append_cv_results(df, result_cv_LC, seed, 'LC')
    df = append_testset_results(df, result_ts_LC, seed, 'LC')
    
    
    df_class_acc = pd.DataFrame(columns = ['Seed', 'Dataset'])
    
    df_class = create_per_class_acc_df(result_ts_70LC[5], seed, '70LC')
    df_class = df_class.append(create_per_class_acc_df(result_ts_35gseLC[5], seed, '35_gseLC'))
    df_class = df_class.append(create_per_class_acc_df(result_ts_70LCgse[5], seed, '70LC_GSE'))
    df_class = df_class.append(create_per_class_acc_df(result_ts_70gseLC[5], seed, '70_gseLC'))
    df_class = df_class.append(create_per_class_acc_df(result_ts_LC[5], seed, 'LC'))

    
    return (df, df_class)


family_label = '/research/rgs01/home/clusterHome/qtran/Semisupervised_Learning/python/raw_data/GSE90496_methylation_family_label.csv'

y = pd.read_csv(family_label, header=0, names=['labels'])
###create a labelencoder to transform str classes to integer classes 
###and use the labelencoder to reverse transform to get the str labels throughout
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(y['labels'])
###
    
###calculate class weights to let the model to pay attention to the under-represented classes
cls_weights = class_weight.compute_class_weight("balanced", classes=np.unique(integer_encoded), y=integer_encoded)
cls_weight_dict = {}
for cls in np.unique(integer_encoded):
    cls_weight_dict[cls] = cls_weights[cls].round(3)
    

tic = time.process_time()
print("Read in beta file")
beta_file = '/research/rgs01/home/clusterHome/qtran/Semisupervised_Learning/python/raw_data/beta_1104_validation_allRef_filtered_noNA.csv'
beta = pd.read_csv(beta_file, index_col=0)
print(beta.shape)
print(beta.iloc[0:5, 0:10])
#beta.drop(beta.filter(regex="Unnamed"),axis=1, inplace=True)
print("Finish dropping column")
toc = time.process_time()
print(beta.shape)
print(toc-tic)


n_split = 3
n_repeat = 5
rand = 123456
rkfold = RepeatedStratifiedKFold(n_splits=n_split, n_repeats=n_repeat, random_state=rand)
for seed in [1, 2, 20, 40, 80, 160, 320]:
    print("start seed=", seed)
    result = evaluate(seed=seed, beta=beta, rkfold = rkfold, encoder=label_encoder, cls_weight = cls_weight_dict, sd_cutoff=0.3)
    output_file = '/research/rgs01/home/clusterHome/qtran/Semisupervised_Learning/python/result_NN_balanced_LCfamily_seed{}_GSE109379_rand{}.csv'.format(seed, rand)
    output_per_family_file ='/research/rgs01/home/clusterHome/qtran/Semisupervised_Learning/python/output_acc_perFamily/result_NN_balanced_acc_perLCFamily_seed{}_GSE109379_rand{}.csv'.format(seed, rand)
    result[0].to_csv(output_file, index=False)
    result[1].to_csv(output_per_family_file, index=False)
    print('Result saved to', output_file)