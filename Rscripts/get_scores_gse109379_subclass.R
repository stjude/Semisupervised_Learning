library(ssc)
library(glmnet)
library(data.table)
library(doParallel)
no_cores <- detectCores() - 1  
registerDoParallel(cores=no_cores)

load("/Volumes/qtran/Semisupervised_Learning/best.setred.svm.subClass.RData")

#source("/Volumes/qtran/Semisupervised_Learning/predict_setred_prob.R")
sorted_xitest = read.table("/Volumes/qtran/Semisupervised_Learning/raw_data/GSE109379_top5K_sorted_input.txt", header=TRUE)
gse_label = read.csv("/Volumes/qtran/Semisupervised_Learning/processed_data/GSE109379_old_vs_selfKNN_labels.csv", header=TRUE)

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
pred_class <- predict(m.setred, as.matrix(sorted_xitest))

source("/rgs01/home/clusterHome/qtran/Semisupervised_Learning/Rscripts/predict_setred_prob.R")
source("/Volumes/qtran/Semisupervised_Learning/Rscripts/predict_setred_prob.R")

pred_iscore <- predict.setred.prob(m.setred, as.matrix(sorted_xitest))

load("/Volumes/qtran/Semisupervised_Learning/calibration_model_subclass.RData")
best.lambda = read.table("/Volumes/qtran/Semisupervised_Learning/lambda_for_calibration_model_subclass.txt", header=FALSE)

gse_calb = predict(fit, pred_iscore, s = best.lambda$V1, type = "response")
gse_calb = as.data.frame(gse_calb)
pred_class = data.frame(pred_class)
rownames(pred_class) = rownames(sorted_xitest)
gse_label$setred.label = pred_class$pred_class[match(gse_label$GSM.ID, substr(rownames(pred_class), 0, 10))]
gse_label$setred.label = pred_class$pred_class[match(gse_label$Sample, rownames(pred_class))]

output = as.data.frame(t(pred_class))
max_score <- apply(gse_calb, 1, max)
names(max_score) = names(output)
output = rbind(output, as.data.frame(t(max_score)))
output = rbind(output, "")
output = rbind(output, t(gse_calb))
rownames(output)[2:3] = c("pred_calb_score", "")

gse_label$setred.max.score = max_score
gse_label$Samples = names(max_score)
gse_label = gse_label[, c('Samples', 'mclass', 'title.class', 'GSM.ID', 'setred.label', 'setred.max.score')]
write.csv(gse_label, "/rgs01/home/clusterHome/qtran/Semisupervised_Learning/processed_data/GSE109379_old_vs_setred_labels.csv")
write.csv(output, "/rgs01/home/clusterHome/qtran/Semisupervised_Learning/processed_data/GSE109379_setred_output.csv")
write.csv(output, "/Volumes/qtran/Semisupervised_Learning/processed_data/GSE109379_setred_output_24Jan2022.csv")
write.csv(gse_label, "/Volumes/qtran/Semisupervised_Learning/processed_data/GSE109379_old_vs_setred_24Jan2022.csv")

