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
    test_sample_weights = class_weight.compute_sample_weight(cls_weight, y_test)
    acc=accuracy_score(y_test,y_pred).round(3)
    bal_acc=balanced_accuracy_score(y_test,y_pred, sample_weight = test_sample_weights).round(3)
    rec=recall_score(y_test,y_pred, average='weighted', zero_division=0).round(3)
    prec=precision_score(y_test,y_pred, average='weighted', zero_division=0).round(3)
    f1=f1_score(y_test,y_pred, average='weighted').round(3)
    per_class_acc = accuracy_per_class(y_test, y_pred)
    return (acc, bal_acc, rec, prec, f1, per_class_acc)

def evaluate_against_cv_test_set(mod, x_test, y_test, cls_weight):
    y_pred = mod.predict(x_test)
    test_sample_weights = class_weight.compute_sample_weight(cls_weight, y_test)
    acc=accuracy_score(y_test,y_pred).round(3)
    bal_acc=balanced_accuracy_score(y_test,y_pred, sample_weight = test_sample_weights).round(3)
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
       
    file35 = '/research/rgs01/home/clusterHome/qtran/Semisupervised_Learning/python/data_family/seed{}_35perc.csv'.format(seed)
    file70='/research/rgs01/home/clusterHome/qtran/Semisupervised_Learning/python/data_family/seed{}_70perc.csv'.format(seed)
    file70HC= '/research/rgs01/home/clusterHome/qtran/Semisupervised_Learning/python/data_family/seed{}_70percHC.csv'.format(seed)
    
    file35_GSE='/research/rgs01/home/clusterHome/qtran/Semisupervised_Learning/python/data_family/seed{}_35perc_gse109379.csv'.format(seed)
    file35_GSEHC ='/research/rgs01/home/clusterHome/qtran/Semisupervised_Learning/python/data_family/seed{}_35perc_HCgse109379.csv'.format(seed)
    file70_GSE = '/research/rgs01/home/clusterHome/qtran/Semisupervised_Learning/python/data_family/seed{}_70perc_gse109379.csv'.format(seed)
    
    file70HC_GSE = '/research/rgs01/home/clusterHome/qtran/Semisupervised_Learning/python/data_family/seed{}_70HC_gse109379.csv'.format(seed)
    file70_GSEHC = '/research/rgs01/home/clusterHome/qtran/Semisupervised_Learning/python/data_family/seed{}_70perc_gse109379HC.csv'.format(seed)
    file70HC_GSEHC = '/research/rgs01/home/clusterHome/qtran/Semisupervised_Learning/python/data_family/seed{}_70HC_gse109379HC.csv'.format(seed)
    
    fileHoldOut = '/research/rgs01/home/clusterHome/qtran/Semisupervised_Learning/python/data_family/seed{}_holdOutTest.csv'.format(seed)
    #########
    print('\tReading', file35)
    X35, y35 = get_beta_in_select_file(probe_data, file35)
    print('\tReading', file70)
    X70, y70 = get_beta_in_select_file(probe_data, file70)
    print('\tReading', file70HC)
    X70HC, y70HC = get_beta_in_select_file(probe_data, file70HC)
    
    ##
    print('\tReading', file35_GSE)
    X35gse, y35gse = get_beta_in_select_file(probe_data, file35_GSE)
    print('\tReading', file35_GSEHC)
    X35gseHC, y35gseHC = get_beta_in_select_file(probe_data, file35_GSEHC)
    print('\tReading', file70_GSE)
    X70gse, y70gse = get_beta_in_select_file(probe_data, file70_GSE)
    ##
    print('\tReading', file70HC_GSE)
    X70HCgse, y70HCgse = get_beta_in_select_file(probe_data, file70HC_GSE)
    print('\tReading', file70_GSEHC)
    X70gseHC, y70gseHC = get_beta_in_select_file(probe_data, file70_GSEHC)
    print('\tReading', file70HC_GSEHC)
    XHC, yHC = get_beta_in_select_file(probe_data, file70HC_GSEHC)
    
    
    print('\tReading', fileHoldOut)
    xitest, yitest = get_beta_in_select_file(probe_data, fileHoldOut)
    ###################
    
    
    print('2. Evaluate model with 35% data')
    print('\tcross validate', model, 'with', cv)
    mod35, result_cv_35 = cross_validate_withSD(model, X35, y35, cv, sd_cutoff = 0.3, cls_weight = cls_weight_dict)
    print('\tvalidate against test set')
    result_ts_35 = evaluate_against_test_set(mod35, X35, y35, xitest, yitest, cls_weight=cls_weight_dict)
          
    print('3. Evaluate model with 70% data')
    print('\tcross validate', model, 'with', cv)
    mod70, result_cv_70 = cross_validate_withSD(model, X70, y70, cv, sd_cutoff = 0.3, cls_weight = cls_weight_dict)
    print('\tvalidate against test set')
    result_ts_70 = evaluate_against_test_set(mod70, X70, y70, xitest, yitest, cls_weight=cls_weight_dict)

    print('4. Evaluate model with 70% HC')    
    print('\tcross validate', model, 'with', cv)
    mod70HC, result_cv_70HC = cross_validate_withSD(model, X70HC, y70HC, cv, sd_cutoff = 0.3, cls_weight = cls_weight_dict)
    print('\tvalidate against test set')
    result_ts_70HC = evaluate_against_test_set(mod70HC, X70HC, y70HC, xitest, yitest, cls_weight=cls_weight_dict)
    
    print('5. Evaluate model with 35% GSE')   
    mod35gse, result_cv_35gse = cross_validate_withSD(model, X35gse, y35gse, cv, sd_cutoff = 0.3, cls_weight = cls_weight_dict)
    print('\tvalidate against test set')
    result_ts_35gse = evaluate_against_test_set(mod35gse, X35gse, y35gse, xitest, yitest, cls_weight=cls_weight_dict)
    
    print('6. Evaluate model with 35% GSE HC')   
    mod35gseHC, result_cv_35gseHC = cross_validate_withSD(model, X35gseHC, y35gseHC, cv, sd_cutoff = 0.3, cls_weight = cls_weight_dict)
    print('\tvalidate against test set')
    result_ts_35gseHC = evaluate_against_test_set(mod35gseHC, X35gseHC, y35gseHC, xitest, yitest, cls_weight=cls_weight_dict)
    
    print('7. Evaluate model with 70% GSE')   
    mod70gse, result_cv_70gse = cross_validate_withSD(model, X70gse, y70gse, cv, sd_cutoff = 0.3, cls_weight = cls_weight_dict)
    print('\tvalidate against test set')
    result_ts_70gse = evaluate_against_test_set(mod70gse, X70gse, y70gse, xitest, yitest, cls_weight=cls_weight_dict)
    
    print('8. Evaluate model with 70% GSE HC')   
    mod70gseHC, result_cv_70gseHC = cross_validate_withSD(model, X70gseHC, y70gseHC, cv, sd_cutoff = 0.3, cls_weight = cls_weight_dict)
    print('\tvalidate against test set')
    result_ts_70gseHC = evaluate_against_test_set(mod70gseHC, X70gseHC, y70gseHC, xitest, yitest, cls_weight=cls_weight_dict)
    
    print('9. Evaluate model with 70% HC GSE')   
    mod70HCgse, result_cv_70HCgse = cross_validate_withSD(model, X70HCgse, y70HCgse, cv, sd_cutoff = 0.3, cls_weight = cls_weight_dict)
    print('\tvalidate against test set')
    result_ts_70HCgse = evaluate_against_test_set(mod70HCgse, X70HCgse, y70HCgse, xitest, yitest, cls_weight=cls_weight_dict)
    
    print('10. Evaluate model with 70% HC GSE HC')   
    modHC, result_cv_HC = cross_validate_withSD(model, XHC, yHC, cv, sd_cutoff = 0.3, cls_weight = cls_weight_dict)
    print('\tvalidate against test set')
    result_ts_HC = evaluate_against_test_set(modHC, XHC, yHC, xitest, yitest, cls_weight=cls_weight_dict)
    #########
    
        
    print('6. Store results.')
    df = pd.DataFrame(data=[], columns=['Seed','Dataset','Validation','Metric','Value'])
    df = append_cv_results(df, result_cv_35, seed, '35')
    df = append_testset_results(df, result_ts_35, seed, '35')
    
    df = append_cv_results(df, result_cv_70, seed, '70')
    df = append_testset_results(df, result_ts_70, seed, '70')
    
    df = append_cv_results(df, result_cv_70HC, seed, '70HC')
    df = append_testset_results(df, result_ts_70HC, seed, '70HC')
    
    df = append_cv_results(df, result_cv_35gse, seed, '35_GSE')
    df = append_testset_results(df, result_ts_35gse, seed, '35_GSE')
    
    df = append_cv_results(df, result_cv_35gseHC, seed, '35_gseHC')
    df = append_testset_results(df, result_ts_35gseHC, seed, '35_gseHC')
    
    df = append_cv_results(df, result_cv_70gse, seed, '70_GSE')
    df = append_testset_results(df, result_ts_70gse, seed, '70_GSE')
    
    df = append_cv_results(df, result_cv_70HCgse, seed, '70HC_GSE')
    df = append_testset_results(df, result_ts_70HCgse, seed, '70HC_GSE')
    df = append_cv_results(df, result_cv_70gseHC, seed, '70_gseHC')
    df = append_testset_results(df, result_ts_70gseHC, seed, '70_gseHC')
    df = append_cv_results(df, result_cv_HC, seed, 'HC')
    df = append_testset_results(df, result_ts_HC, seed, 'HC')
    
    df_class = create_per_class_acc_df(result_ts_35[5], seed, '35')
    
    df_class = df_class.append(create_per_class_acc_df(result_ts_70[5], seed, '70'))
    df_class = df_class.append(create_per_class_acc_df(result_ts_70HC[5], seed, '70HC'))
    
    df_class = df_class.append(create_per_class_acc_df(result_ts_35gse[5], seed, '35GSE'))
    df_class = df_class.append(create_per_class_acc_df(result_ts_35gseHC[5], seed, '35_gseHC'))
    df_class = df_class.append(create_per_class_acc_df(result_ts_70gse[5], seed, '70_GSE'))
    
    df_class = df_class.append(create_per_class_acc_df(result_ts_70HCgse[5], seed, '70HC_GSE'))
    df_class = df_class.append(create_per_class_acc_df(result_ts_70gseHC[5], seed, '70_gseHC'))
    df_class = df_class.append(create_per_class_acc_df(result_ts_HC[5], seed, 'HC'))
       
    return (df, df_class)

def run_all_seeds(rand):
    model = RandomForestClassifier(n_estimators = 50, max_depth=100)
    cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=5, random_state=rand)
    for seed in [1]:
        result = evaluate(seed, beta, model, cv)
        output_file = '/research/rgs01/home/clusterHome/qtran/Semisupervised_Learning/python/output/result_RF_family_seed{}_GSE109379_rand{}.csv'.format(seed, rand)
        output_per_class_file = '/research/rgs01/home/clusterHome/qtran/Semisupervised_Learning/python/output_acc_perFamily/result_RF_acc_perFamily_seed{}_GSE109379_rand{}.csv'.format(seed, rand)
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