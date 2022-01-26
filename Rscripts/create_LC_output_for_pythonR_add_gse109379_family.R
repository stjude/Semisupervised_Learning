library(data.table)
library(doMC)
registerDoMC(cores=8)
dir = "/research/rgs01/home/clusterHome/qtran/Semisupervised_Learning"
#dir = "/Volumes/qtran/Semisupervised_Learning"
setwd(dir)

ySSL = read.csv("./processed_data/family_results/Calibrated_Trans_scores_MCF_long.csv", header=TRUE)
ySSL$pred_family = trimws(ySSL$pred_family)
#gse_xitest = read.csv("./python/raw_data/beta_1104_validation_allRef_filtered_noNA.csv", sep=",")
gse_xitest = read.table("./raw_data/GSE109379_top5K_sorted_input.txt", header=TRUE)
gse_y = read.csv("./processed_data/GSE109379_old_vs_setred_family_labels.csv", header=TRUE)
gse_y$setred.label = trimws(gse_y$setred.label)

gse_xitest$y = gse_y$setred.label[match(rownames(gse_xitest), gse_y$Samples)]
gse_hc_samples = gse_y$Samples[gse_y$setred.max.score > 0.73]  

gse_xitest_hc = gse_xitest[rownames(gse_xitest) %in% gse_hc_samples,]
gse_xitest_lc = gse_xitest[!(rownames(gse_xitest) %in% gse_hc_samples),]

get_high_conf_samples = function(labelMaxScore, xtrain, ytrain, thres = NULL){
  comb_data = as.data.frame(cbind(xtrain, ytrain))
  colnames(comb_data)[ncol(comb_data)] = "y"
  comb_data$y[which(is.na(comb_data$y))] = 
    labelMaxScore$pred_family[match(rownames(comb_data)[which(is.na(comb_data$y))] ,labelMaxScore$Sample)]
  comb_data$y = trimws(comb_data$y)
  #print(comb_data[1, 5070:ncol(comb_data)])
  if(!is.null(thres)) {
    no.samples = labelMaxScore$Sample[labelMaxScore$pred_score < thres] # excluded samples (LC sample)
    comb_data = comb_data[!(rownames(comb_data) %in% no.samples),] #exclude LC samples
  }
  return(comb_data)
}

get_low_conf_samples = function(labelMaxScore, xtrain, ytrain, thres = NULL){
  comb_data = as.data.frame(cbind(xtrain, ytrain))
  colnames(comb_data)[ncol(comb_data)] = "y"
  comb_data$y[which(is.na(comb_data$y))] = 
    labelMaxScore$pred_family[match(rownames(comb_data)[which(is.na(comb_data$y))] ,labelMaxScore$Sample)]
  comb_data$y = trimws(comb_data$y)
  #print(comb_data[1, 5070:ncol(comb_data)])
  if(!is.null(thres)) {
    low.samples = labelMaxScore$Sample[labelMaxScore$pred_score < thres] # 
    comb_data = comb_data[rownames(comb_data) %in% low.samples,] # LC samples
  }
  return(comb_data)
}

seeds = c("seed1", "seed2", "seed20", "seed40", "seed80", "seed160", "seed320")
#seeds = c("seed320")
for (i in seeds){
  print(i)
  bdir = paste0(dir, "/python/data_family/", i)
  f1 = paste0(dir, "/processed_data/family_results/", i, "_alldata_sets.RData")
  load(f1)
  xtrain = x$xtrain
  ytrain = trimws(x$ytrain)
  ytrain = as.matrix(ytrain)
  
  xitest = x$xitest
  yitest = trimws(x$yitest)
  yitest = as.matrix(yitest)

  xttest = x$xttest
  yttest = trimws(x$yttest)
  yttest = as.matrix(yttest)

  tra.na.idx = x$tra.na.idx 
  
  keep.idx <- setdiff(1:length(ytrain), tra.na.idx)
  xtrain_nonNA = data.frame(xtrain[keep.idx,])
  xtrain_nonNA = as.matrix(xtrain_nonNA)
  ytrain_nonNA = as.matrix(ytrain[keep.idx])
  
  train_data = as.data.frame(cbind(xtrain_nonNA, ytrain_nonNA))
  colnames(train_data)[ncol(train_data)] = 'y'
  
  test_data = as.data.frame(cbind(xitest, yitest))
  colnames(test_data)[ncol(test_data)] = 'y'
  
  train_plus_LCgse = rbind(train_data, gse_xitest_lc)
  write.csv(train_plus_LCgse, file = paste0(bdir, "_35perc_LCgse109379.csv"))
  print("Done File4!")
  
  ySSL_seed = ySSL[ySSL$seed == paste0(i, "_all"), ]
  ###get the label with max scores
  labelseed = merge(aggregate(pred_score ~ Sample, data=ySSL_seed, max), ySSL_seed, all.x=T)
  
  nothres = get_high_conf_samples(labelseed, xtrain, ytrain, thres=NULL)
  nothres_plus_gseLC = rbind(nothres, gse_xitest_lc)
  write.csv(nothres_plus_gseLC, file = paste0(bdir, "_70perc_gse109379LC.csv"))
  print("Done File6!")

  thres_list = get_low_conf_samples(labelseed, xttest, yttest, 0.73)
  LC_70 = rbind(train_data, thres_list)
  write.csv(LC_70, file = paste0(bdir, "_70percLC.csv"))
  print("Done File7!")
  
  xLC_gse = rbind(LC_70, gse_xitest)
  write.csv(xLC_gse, file = paste0(bdir, "_70LC_gse109379.csv")) ###35% HC pseudo label + all gse pseudo-label
  print("Done File8!")
  
  xLC_gseLC = rbind(LC_70, gse_xitest_lc)
  write.csv(xLC_gseLC, file = paste0(bdir, "_70LC_gse109379LC.csv"))
  print("Done File10!")
  
}

