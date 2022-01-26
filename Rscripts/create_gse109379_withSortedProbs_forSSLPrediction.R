library(ssc)
library(data.table)
library(doParallel)
no_cores <- detectCores() - 1  
registerDoParallel(cores=no_cores)

#load("/Volumes/qtran/Semisupervised_Learning/processed_data/class_results/seed20_allmodelsALL.RData")
load("/rgs01/home/clusterHome/qtran/Semisupervised_Learning/processed_data/class_results/seed20_allmodelsALL.RData")

#load("/rgs01/home/clusterHome/qtran/Semisupervised_Learning/label_data_x.RData")
#load("/rgs01/home/clusterHome/qtran/Semisupervised_Learning/label_data_y.RData")
#remove(x)
source("/Volumes/qtran/Semisupervised_Learning/predict_setred_prob.R")
source("/rgs01/home/clusterHome/qtran/Semisupervised_Learning/Rscripts/predict_setred_prob.R")

load("/rgs01/home/clusterHome/qtran/Semisupervised_Learning/data_sets.RData")
load("/rgs01/home/clusterHome/qtran/Semisupervised_Learning/processed_data/family_results/seed20_alldata_sets.RData")

xtrain = x$xtrain
ytrain = x$ytrain

gse109379 = fread("/rgs01/home/clusterHome/qtran/Capper_validation_set/processed_data/beta_1104Capper_validation_filtered.txt")
gse109379 = fread("/Volumes/qtran/Capper_validation_set/processed_data/beta_1104Capper_validation_filtered.txt")

gse_label = read.csv("/rgs01/home/clusterHome/qtran/Capper_validation_set/raw_data/GSE109379_DKFZ_key-1-21-19-clean.csv", header=TRUE)

top_gse109379 = as.data.frame(gse109379[gse109379$V1 %in% colnames(xtrain),])
rownames(top_gse109379) = top_gse109379$V1
top_gse109379 = top_gse109379[,-1]

gse_xitest = t(top_gse109379) 
  
get_features_prediction = function(beta, train_data){
  top_beta = as.data.frame(beta[beta$V1 %in% colnames(train_data),])
  rownames(top_beta) = top_beta$V1
  top_beta = top_beta[,-1]
  t_beta = t(top_beta) 
  
  missing_probes = which(!(colnames(train_data) %in% colnames(t_beta)))
  if (length(missing_probes) != 0){
    missing_data = train_data[, missing_probes]
    avg_data = apply(missing_data, 2, mean)
    print(missing_probes)
    temp = data.frame(t(avg_data))
    temp2 = temp[rep(seq_len(nrow(temp)), each=nrow(t_beta)), ]###repeat each row of temp data for 64 times (the #of row)
    rownames(temp2) = rownames(t_beta)
    
    temp2 = cbind(t_beta, temp2) 
    
    xitest = as.matrix(temp2[ , colnames(train_data)])
  } else{
    print("HERE")
    xitest = t_beta[ , colnames(train_data)]
  }
  return(xitest)
}
gse_xitest = get_features_prediction(gse109379, xtrain)
sorted_xitest = gse_xitest[ , colnames(xtrain)]

####IT'S PERTINENT TO SORT THE PROBES TO MATCH THE TRAINING DATA#####
write.table(sorted_xitest, file = "/Volumes/qtran/Semisupervised_Learning/raw_data/GSE109379_top5K_sorted_input.txt")

write.table(sorted_xitest, file = "/research/rgs01/home/clusterHome/qtran/Semisupervised_Learning/raw_data/GSE109379_top5K_sorted_input.txt")

sorted_xitest = read.table("/Volumes/qtran/Semisupervised_Learning/raw_data/GSE109379_top5K_sorted_input.txt", header=TRUE)

