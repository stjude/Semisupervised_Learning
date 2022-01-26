library(ssc)
library(data.table)
library(doParallel)
no_cores <- detectCores() - 1  
registerDoParallel(cores=no_cores)
load("/Volumes/qtran/Semisupervised_Learning/processed_data/class_results/seed20_allmodelsALL.RData")
load("/rgs01/home/clusterHome/qtran/Semisupervised_Learning/processed_data/class_results/seed20_allmodelsALL.RData")

#load("/rgs01/home/clusterHome/qtran/Semisupervised_Learning/label_data_x.RData")
#load("/rgs01/home/clusterHome/qtran/Semisupervised_Learning/label_data_y.RData")
#m.self = x$m.selft1
m.setred = x$m.setred2
source("/Volumes/qtran/Semisupervised_Learning/predict_setred_prob.R")
source("/rgs01/home/clusterHome/qtran/Semisupervised_Learning/predict_setred_prob.R")

load("/rgs01/home/clusterHome/qtran/Semisupervised_Learning/processed_data/class_results/seed20_alldata_sets.RData")
load("/Volumes/qtran/Semisupervised_Learning/processed_data/class_results/seed20_alldata_sets.RData")

xtrain = x$xtrain
ytrain = x$ytrain
rm(x)
act = fread("/rgs01/home/clusterHome/qtran/Solid_tumor/ACT/processed_data/ACT-withControl-FunNorm-850K-Beta-cgkeep-B3-10-16-18.txt")
rms = fread("/Volumes/qtran/Solid_tumor/RMS/processed_data/RMS-FunNorm-850K-Beta-NoOutliers-cgkeep-B3-10-25-18.txt")
gse109379 = fread("/rgs01/home/clusterHome/qtran/Capper_validation_set/processed_data/beta_1104Capper_validation_filtered.txt")

#####
##The ACT data set is 850K and so do not have all the probes in the 450K form the xtrain
##Include the missing probes and use the mean values of all samples for those probes.
####IT'S PERTINENT TO SORT THE PROBES TO MATCH THE TRAINING DATA#####

get_features_prediction = function(beta, train_data){
  top_beta = as.data.frame(beta[beta$V1 %in% colnames(train_data),])
  rownames(top_beta) = top_beta$V1
  top_beta = top_beta[,-1]
  t_beta = t(top_beta) 
  
  missing_probes = which(!(colnames(train_data) %in% colnames(t_beta)))
  print(head(missing_probes))
  if (length(missing_probes) >0 ){
    missing_data = train_data[, missing_probes]
    avg_data = apply(missing_data, 2, mean)
    temp = data.frame(t(avg_data))
    temp2 = temp[rep(seq_len(nrow(temp)), each=nrow(t_beta)), ]###repeat each row of temp data for 64 times (the #of row)
    rownames(temp2) = rownames(t_beta)
    
    temp2 = cbind(t_beta, temp2) 
    
    xitest = as.matrix(temp2[ , colnames(train_data)])
  }else{
    xitest = as.matrix(t_beta[ , colnames(train_data)])
  }
  return(xitest)
}
gse_xitest[is.na(gse_xitest)] = 0.5

###check for NA values in the data
for(i in 1:ncol(top_beta)){
  if(is.na(top_beta[,i])){
    
  }
  
}

xitest = get_features_prediction(act, xtrain)
rms_xitest = get_features_prediction(rms, xtrain)
write.table(gse_xitest, file = "/research/rgs01/home/clusterHome/qtran/Semisupervised_Learning/raw_data/GSE109379_top5K_sorted_input.txt")

###Set up learner parameters 
learner <- e1071::svm
learner.pars <- list(type = "C-classification", kernel="radial", 
                     probability = TRUE, scale = TRUE)
pred <- function(m, x){
  r <- predict(m, x, probability = TRUE)
  prob <- attr(r, "probabilities")
  return(prob)
}

#######predict as inductive testing######
pred_class <- predict(m.setred, xitest)
rms_class = predict(m.setred, rms_xitest)
gse_class = predict(m.setred, gse_xitest)
#ditest <- as.matrix(proxy::dist(x = xitest, y = xtrain, method = "euclidean", by_rows = TRUE))
#pred_class_knn = predict(m.self, ditest[, m.self$instances.index])

source("/rgs01/home/clusterHome/qtran/Semisupervised_Learning/predict_setred_prob.R")
#ource("/Volumes/qtran/Semisupervised_Learning/predictOneNN.R")

pred_iscore <- predict.setred.prob(m.setred, xitest)
rms_score = predict.setred.prob(m.setred, rms_xitest)
#pred_iscore_knn = predProb.OneNN(m.self, ditest[, m.self$instances.index])


create_output = function(pred_score, pred_class, outfile){
  pred_class = data.frame(pred_class)
  rownames(pred_class) = rownames(pred_score)

  output = as.data.frame(t(pred_class))
  max_score <- apply(pred_score, 1, max)
  names(max_score) = names(output)
  output = rbind(output, as.data.frame(t(max_score)))
  output = rbind(output, "")
  output = rbind(output, as.data.frame(t(pred_score)))
  rownames(output)[2:3] = c("pred_score", "")
  print(max(max_score))
  write.csv(output, outfile)
}

create_output(pred_iscore, pred_class, outfile = "/Volumes/qtran/Semisupervised_Learning/processed_data/ACT_setred_predicted_raw.csv")
create_output(rms_score, rms_class, outfile = "/Volumes/qtran/Semisupervised_Learning/processed_data/RMS_setred_predicted_raw.csv")

#########calibrate the scores#####
load("/Volumes/qtran/Semisupervised_Learning/calibration_model.RData")
best.lambda = read.table("/Volumes/qtran/Semisupervised_Learning/lambda_for_calibration_model.txt", header=FALSE)

act_calb = predict(fit, pred_iscore, s = best.lambda$V1, type = "response")
rms_calb = predict(fit, rms_score, s=best.lambda$V1, type = "response")

create_output(as.data.frame(act_calb), pred_class, outfile = "/Volumes/qtran/Semisupervised_Learning/processed_data/ACT_setred_predicted_calibrated.csv")
create_output(as.data.frame(rms_calb), rms_class, outfile = "/Volumes/qtran/Semisupervised_Learning/processed_data/RMS_setred_predicted_calibrated.csv")

