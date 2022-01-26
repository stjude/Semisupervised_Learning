library(randomForest)
library(caret)
library(e1071)
#library(doParallel)
library(MLmetrics)
library(doMC)
registerDoMC(cores=18)
library(data.table)
setwd("/research/rgs01/home/clusterHome/qtran/Semisupervised_Learning/")
#load("./label_data_x.RData")
load("./class_label_data_y.RData")

beta = fread("/research/rgs01/home/clusterHome/qtran/Capper_reference_set/beta_GSE90496_cgkeep.txt")
beta = as.data.frame(beta)
rownames(beta) = beta$V1
beta = beta[,-1]
tbeta = t(beta)
###unlike save(), saveRDS() saved the object without its name and are smaller than using write.csv()
saveRDS(tbeta, file = "/research/rgs01/home/clusterHome/qtran/Capper_reference_set/GSE90496_beta_cgkeep_transposed.RDS")
rm(beta)
label_data_y = as.factor(label_data_y)
labelfreq = table(label_data_y)

set.seed(1234)
rf = foreach(ntree=rep(100,20), .combine=combine, .packages='randomForest') %dopar%
  randomForest(tbeta,y=label_data_y,ntree=ntree,do.trace=1,mtry=654,strata=label_data_y,
               importance=TRUE,sampsize=rep(min(labelfreq),dim(labelfreq)))

vals = importance(rf,type=1)

