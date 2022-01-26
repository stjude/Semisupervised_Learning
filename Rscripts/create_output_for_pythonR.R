library(R.matlab)
library(data.table)
library(doMC)
registerDoMC(cores=10)
dir = "/research/rgs01/home/clusterHome/qtran/Semisupervised_Learning"
#dir = "/Volumes/qtran/Sesmisupervised_Learning"
setwd(dir)

ySSL = read.csv("./processed_data/class_results/Calibrated_Trans_scores_long.csv", header=TRUE)

get_high_conf_samples = function(labelMaxScore, xtrain, ytrain, thres = NULL){
  comb_data = as.data.frame(cbind(xtrain, ytrain))
  colnames(comb_data)[ncol(comb_data)] = "y"
  comb_data$y[which(is.na(comb_data$y))] = 
    labelMaxScore$pred_label[match(rownames(comb_data)[which(is.na(comb_data$y))] ,labelMaxScore$Sample)]
  comb_data$y = trimws(comb_data$y)
  #print(comb_data[1, 5070:ncol(comb_data)])
  if(!is.null(thres)) {
    no.samples = labelMaxScore$Sample[labelMaxScore$pred_score < thres]
    comb_data = comb_data[!(rownames(comb_data) %in% no.samples),]
  }
  return(comb_data)
}

seeds = c("seed1", "seed2", "seed20", "seed40", "seed80", "seed160", "seed320")
#seeds = c("seed320")
for (i in seeds){
  bdir = paste0(dir, "/python/data/", i)
  f1 = paste0(dir, "/processed_data/class_results/", i, "_alldata_sets.RData")
  load(f1)
  xtrain = x$xtrain
  ytrain = as.matrix(x$ytrain)
  xitest = x$xitest
  yitest = as.matrix(x$yitest)
  xttest = x$xttest
  yttest = as.matrix(x$yttest)
  tra.na.idx = x$tra.na.idx 
  
  keep.idx <- setdiff(1:length(ytrain), tra.na.idx)
  xtrain_nonNA = data.frame(xtrain[keep.idx,])
  xtrain_nonNA = as.matrix(xtrain_nonNA)
  ytrain_nonNA = as.matrix(ytrain[keep.idx])
  
  train_data = as.data.frame(cbind(xtrain_nonNA, ytrain_nonNA))
  colnames(train_data)[ncol(train_data)] = 'y'
  
  test_data = as.data.frame(cbind(xitest, yitest))
  colnames(test_data)[ncol(test_data)] = 'y'
  
  write.csv(train_data, file = paste0(bdir, "_35perc_train.csv"))
  write.csv(test_data, file = paste0(bdir, "_holdOutTest.csv"))
  
  ySSL_seed = ySSL[ySSL$seed == paste0(i, "_all"), ]
  ###get the label with max scores
  #aggregate(pred_score ~ Sample, data=yttest_seed800, max)
  
  labelseed = merge(aggregate(pred_score ~ Sample, data=ySSL_seed, max), ySSL_seed, all.x=T)
  
  nothres = get_high_conf_samples(labelseed, xtrain, ytrain, thres=NULL)
  
  write.csv(nothres, file = paste0(bdir, "_70perc_SSLabel.csv"))
  
  thres_list = get_high_conf_samples(labelseed, xtrain, ytrain, 0.718)
  
  write.csv(thres_list, file = paste0(bdir, "_SSLabel_HC.csv"))
}

