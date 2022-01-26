import scipy.io
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score,f1_score,precision_score,recall_score 
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def get_beta_in_select_file(beta, filename):
    X_temp, y = process_csv(filename)
    X = beta[ X_temp.index ]
    return X.T, y
 
def process_csv(filename):
    df = pd.read_csv(filename, index_col=0)
    y = df.y
    X = df.drop(columns=['y'])
    print("num class", len(np.unique(y)))
    return (X, y)

def append_result(p, df):
    s = pd.Series(p, index=df.columns)
    return df.append(s, ignore_index=True)

def select_probes(data, sd_cutoff):
    probes = data.T
    probes['STD'] = probes.std(axis=1)
    above_threshold = probes[probes["STD"] > sd_cutoff]
    print("shape in select_probes", above_threshold.T.shape)
    return above_threshold.drop(columns='STD').T

def match_probes(ref_data, data_to_be_matched):
    df = data_to_be_matched.loc[:, ref_data.columns.values]
    return df

def accuracy_per_class(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    classes = np.unique(y_true)
    per_class_accuracies = cm.diagonal()/cm.sum(axis=1)
    per_class_acc_wKeys = {}
    for idx, cls in enumerate(classes):
        per_class_acc_wKeys[cls] = per_class_accuracies[idx]
    return per_class_acc_wKeys

def evaluate_against_test_set(mod, x_train, y_train, x_test, y_test, cls_weight):
    mod.fit(x_train, y_train)
    y_pred = mod.predict(x_test)
    #test_sample_weights = class_weight.compute_sample_weight(cls_weight, y_test)
    acc=accuracy_score(y_test,y_pred).round(3)
    bal_acc=balanced_accuracy_score(y_test,y_pred).round(3)
    rec=recall_score(y_test,y_pred, average='weighted', zero_division=0).round(3)
    prec=precision_score(y_test,y_pred, average='weighted', zero_division=0).round(3)
    f1=f1_score(y_test,y_pred, average='weighted').round(3)
    per_class_acc = accuracy_per_class(y_test, y_pred)
    return (acc, bal_acc, rec, prec, f1, per_class_acc)

def evaluate_against_cv_test_set(mod, x_test, y_test, cls_weight):
    y_pred = mod.predict(x_test)
    #test_sample_weights = class_weight.compute_sample_weight(cls_weight, y_test)
    acc=accuracy_score(y_test,y_pred).round(3)
    bal_acc=balanced_accuracy_score(y_test,y_pred).round(3)
    rec=recall_score(y_test,y_pred, average='weighted', zero_division=0).round(3)
    prec=precision_score(y_test,y_pred, average='weighted', zero_division=0).round(3)
    f1=f1_score(y_test,y_pred, average='weighted').round(3)
    return (acc, bal_acc, rec, prec, f1)

def cross_validate_withSD(mod, data_x, data_y, cv, sd_cutoff, cls_weight):
    '''
    features are selectiving for the training data set after each CV split
    '''
    #cvscores = []
    cv_balanced_scores = []
    fold = 1
    #model = RandomForestClassifier(n_estimators = 50, max_depth=100)
    for train_index,test_index in cv.split(data_x, data_y):
        #print("X_cv shape = ", data_x.shape, "; y_cv shape = ", data_y.shape)
        x_train = select_probes(data_x.iloc[train_index, :], sd_cutoff)
        x_test = match_probes(x_train, data_x.iloc[test_index, :])
        
        y_train,y_test = data_y[train_index], data_y[test_index]
        
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold} ...')
        mod.fit(x_train, y_train)
        scores = evaluate_against_cv_test_set(mod, x_test, y_test, cls_weight)
    
        cv_balanced_scores.append(scores)
        
        print("Scores = ", scores)
        
        fold = fold+1
    return mod, pd.DataFrame(cv_balanced_scores, columns= ['acc', 'bal_acc', 'weighted_recall','weighted_precision', 'weighted_F1'])


def append_cv_results(df, r, seed_name, dataset_name):
    df = append_result([seed_name, dataset_name, 'cross_val', 'accuracy', r['acc'].mean()], df)
    df = append_result([seed_name, dataset_name, 'cross_val', 'balanced_acc', r['bal_acc'].mean()], df)
    df = append_result([seed_name, dataset_name, 'cross_val', 'recall_weighted', r['weighted_recall'].mean()], df)
    df = append_result([seed_name, dataset_name, 'cross_val', 'precision_weighted', r['weighted_precision'].mean()], df)
    return df

def accuracy_per_class(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    classes = np.unique(y_true)
    per_class_accuracies = cm.diagonal()/cm.sum(axis=1)
    per_class_acc_wKeys = {}
    for idx, cls in enumerate(classes):
        per_class_acc_wKeys[cls] = per_class_accuracies[idx]
    return per_class_acc_wKeys

def append_testset_results(df, r, seed_name, dataset_name):
    df = append_result([seed_name, dataset_name, 'vs_testset', 'accuracy', r[0]], df)
    df = append_result([seed_name, dataset_name, 'vs_testset', 'balanced_acc', r[1]], df)
    df = append_result([seed_name, dataset_name, 'vs_testset', 'recall_weighted', r[2]], df)
    df = append_result([seed_name, dataset_name, 'vs_testset', 'precision_weighted', r[3]], df)
    return df

def create_per_class_acc_df(class_acc, seed, dset):
    d = pd.DataFrame(columns = ['Seed', 'Dataset'])
    d = d.append(class_acc, ignore_index=True)
    d['Seed'] = seed
    d['Dataset'] = dset
    return d

def evaluate(seed, probe_data, model, cv, mcf=False):
    print('1. Read files (seed={})'.format(seed)) 
    
    file70LC= '/research/rgs01/home/clusterHome/qtran/Semisupervised_Learning/python/data_family/seed{}_70percLC.csv'.format(seed)  
    file35_GSELC ='/research/rgs01/home/clusterHome/qtran/Semisupervised_Learning/python/data_family/seed{}_35perc_LCgse109379.csv'.format(seed)
    
    file70LC_GSE = '/research/rgs01/home/clusterHome/qtran/Semisupervised_Learning/python/data_family/seed{}_70LC_gse109379.csv'.format(seed)
    file70_GSELC = '/research/rgs01/home/clusterHome/qtran/Semisupervised_Learning/python/data_family/seed{}_70perc_gse109379LC.csv'.format(seed)
    file70LC_GSELC = '/research/rgs01/home/clusterHome/qtran/Semisupervised_Learning/python/data_family/seed{}_70LC_gse109379LC.csv'.format(seed)
    
    fileHoldOut = '/research/rgs01/home/clusterHome/qtran/Semisupervised_Learning/python/data_family/seed{}_holdOutTest.csv'.format(seed)
    #########
    print('\tReading', file70LC)
    X70LC, y70LC = get_beta_in_select_file(probe_data, file70LC)
    
    ##
    print('\tReading', file35_GSELC)
    X35gseLC, y35gseLC = get_beta_in_select_file(probe_data, file35_GSELC)
    ##
    print('\tReading', file70LC_GSE)
    X70LCgse, y70LCgse = get_beta_in_select_file(probe_data, file70LC_GSE)
    print('\tReading', file70_GSELC)
    X70gseLC, y70gseLC = get_beta_in_select_file(probe_data, file70_GSELC)
    print('\tReading', file70LC_GSELC)
    XLC, yLC = get_beta_in_select_file(probe_data, file70LC_GSELC)
    
    print('\tReading', fileHoldOut)
    xitest, yitest = get_beta_in_select_file(probe_data, fileHoldOut)
    ################### 
    print('4. Evaluate model with 70% LC')    
    print('\tcross validate', model, 'with', cv)
    mod70LC, result_cv_70LC = cross_validate_withSD(model, X70LC, y70LC, cv, sd_cutoff = 0.3, cls_weight = cls_weight_dict)
    print('\tvalidate against test set')
    result_ts_70LC = evaluate_against_test_set(mod70LC, X70LC, y70LC, xitest, yitest, cls_weight=cls_weight_dict)
    
    print('6. Evaluate model with 35% GSE LC')   
    mod35gseLC, result_cv_35gseLC = cross_validate_withSD(model, X35gseLC, y35gseLC, cv, sd_cutoff = 0.3, cls_weight = cls_weight_dict)
    print('\tvalidate against test set')
    result_ts_35gseLC = evaluate_against_test_set(mod35gseLC, X35gseLC, y35gseLC, xitest, yitest, cls_weight=cls_weight_dict)
    
    print('8. Evaluate model with 70% GSE LC')   
    mod70gseLC, result_cv_70gseLC = cross_validate_withSD(model, X70gseLC, y70gseLC, cv, sd_cutoff = 0.3, cls_weight = cls_weight_dict)
    print('\tvalidate against test set')
    result_ts_70gseLC = evaluate_against_test_set(mod70gseLC, X70gseLC, y70gseLC, xitest, yitest, cls_weight=cls_weight_dict)
    
    print('9. Evaluate model with 70% LC GSE')   
    mod70LCgse, result_cv_70LCgse = cross_validate_withSD(model, X70LCgse, y70LCgse, cv, sd_cutoff = 0.3, cls_weight = cls_weight_dict)
    print('\tvalidate against test set')
    result_ts_70LCgse = evaluate_against_test_set(mod70LCgse, X70LCgse, y70LCgse, xitest, yitest, cls_weight=cls_weight_dict)
    
    print('10. Evaluate model with 70% LC GSE LC')   
    modLC, result_cv_LC = cross_validate_withSD(model, XLC, yLC, cv, sd_cutoff = 0.3, cls_weight = cls_weight_dict)
    print('\tvalidate against test set')
    result_ts_LC = evaluate_against_test_set(modLC, XLC, yLC, xitest, yitest, cls_weight=cls_weight_dict)

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
    
    df_class = create_per_class_acc_df(result_ts_70LC[5], seed, '70LC')
    #df_class = create_per_class_acc_df(result_ts_35gseLC[5], seed, '35_gseLC')
    df_class = df_class.append(create_per_class_acc_df(result_ts_35gseLC[5], seed, '35_gseLC'))
    
    df_class = df_class.append(create_per_class_acc_df(result_ts_70LCgse[5], seed, '70LC_GSE'))
    df_class = df_class.append(create_per_class_acc_df(result_ts_70gseLC[5], seed, '70_gseLC'))
    df_class = df_class.append(create_per_class_acc_df(result_ts_LC[5], seed, 'LC'))

       
    return (df, df_class)

def run_all_seeds(rand):
    model = RandomForestClassifier(n_estimators = 50, max_depth=100)
    cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=5, random_state=rand)
    for seed in [1, 2, 20, 40, 80, 160, 320]:
        result = evaluate(seed, beta, model, cv)
        output_file = '/research/rgs01/home/clusterHome/qtran/Semisupervised_Learning/python/output/result_RF_LCfamily_seed{}_GSE109379_rand{}.csv'.format(seed, rand)
        output_per_class_file = '/research/rgs01/home/clusterHome/qtran/Semisupervised_Learning/python/output_acc_perFamily/result_RF_acc_perLCFamily_seed{}_GSE109379_rand{}.csv'.format(seed, rand)
        result[0].to_csv(output_file, index=False)
        result[1].to_csv(output_per_class_file, index=False)
        print('Result saved to', output_file)
        
        
        
beta_file = '/research/rgs01/home/clusterHome/qtran/Semisupervised_Learning/python/raw_data/beta_1104_validation_allRef_filtered_noNA.csv'
beta = pd.read_csv(beta_file, index_col=0)

class_label = '/research/rgs01/home/clusterHome/qtran/Semisupervised_Learning/python/raw_data/GSE90496_methylation_family_label.csv'

labels = pd.read_csv(class_label, header=0, names=['y'])
labels.y = [x.strip() for x in labels.y.values]
print(labels)
cls_weights = class_weight.compute_class_weight("balanced", classes=np.unique(labels), y=labels['y'])

cls_weight_dict = {}
i = 0
for cls in np.unique(labels):
    cls_weight_dict[cls] = cls_weights[i].round(3)
    i = i+1

print(cls_weight_dict)

rand = 123456
run_all_seeds(rand)