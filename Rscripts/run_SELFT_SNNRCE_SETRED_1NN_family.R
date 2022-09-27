load("/research/rgs01/home/clusterHome/qtran/Semisupervised_Learning/label_data_x.RData")
load("/research/rgs01/home/clusterHome/qtran/Semisupervised_Learning/family_label_data_y.RData")
source("/research/rgs01/home/clusterHome/qtran/Semisupervised_Learning/Rscripts/supporting_functions.R")
#source("/research/rgs01/home/clusterHome/qtran/Semisupervised_Learning/Rscripts/predictOneNN.R")
###Use 70% of the Capper Reference for training
##first get the index of the labeled data
acc = data.frame("Train_function" = "", "Learner"="", "Seed" = "", "Ind_ACC"="", "Trans_ACC" = "")
main_dir = "/research/rgs01/home/clusterHome/qtran/Semisupervised_Learning_big_private_data"
#main_dir = "/Volumes/qtran/Semisupervised_Learning_big_private_data/"
load(paste0(main_dir, "/processed_data/family_results/seed1_alldata_sets.RData"))

library(caret)
library(ssc)
library(kernlab)
create_class_score_output = function(pred_label, pred_score, truth_label){
  output = data.frame(pred_label)
  output$Sample = names(truth_label)
  output$pred_score = apply(pred_score, 1, max)
  output$truth_label = truth_label
  
  return(output)
}

seeds = c(1, 2, 20, 40, 80, 160, 320)

for (s in seeds){
  print(paste0("Start with seed = ", s))
  
  sub_dir = paste0("seed", s, "_all")
  output_tbl_dir <- file.path(main_dir, "processed_data/family_results", sub_dir)
  output_fig_dir <- file.path(main_dir, "figures/family_results", sub_dir)
  if (!dir.exists(output_tbl_dir)){
    dir.create(output_tbl_dir)
  } else {
    print("Table Dir already exists!")
  }
  if (! dir.exists(output_fig_dir)){
    dir.create(output_fig_dir)
  } else {
    print("Figure Dir already exists!")
  }
  
  
  load(paste0(main_dir, "/processed_data/family_results/seed", s, "_alldata_sets.RData"))
  
  tra.idx = x$tra.idx
  
  #tra.idx = get_sample_n_rows_idx(label_data_x, label_data_y, n=nums$min_class, replace=FALSE)
  ####make train, test data sets#######
  xtrain = NULL
  ytrain = NULL
  xtrain <- x$xtrain
  ytrain <- x$ytrain
  
  ###Use 50% of train instances as unlabeled set
  tra.na.idx <- x$tra.na.idx
  ytrain[tra.na.idx] <- NA
  
  #Use the other 30% of instances for inductive test
 
  xitest <- x$xitest # test instances
  yitest <- x$yitest # classes of instances in xitest
  
  # Use the unlabeled examples for transductive test
  xttest <- x$xttest # transductive test instances 
  yttest <- x$yttest # classes of instances in xttest
  
  #############base line supervised learning classifier SVM##########

  ################ computing distance and kernel matrices
  dtrain <- as.matrix(proxy::dist(x = xtrain, method = "euclidean", by_rows = TRUE)) 
  ditest <- as.matrix(proxy::dist(x = xitest, y = xtrain, method = "euclidean", by_rows = TRUE))
  dttest <- as.matrix(proxy::dist(x = xttest, y = xtrain, method = "euclidean", by_rows = TRUE))
  ###Set up learner parameters 
  learner <- e1071::svm
  learner.pars <- list(type = "C-classification", kernel="radial", 
                       probability = TRUE, scale = TRUE)
  pred <- function(m, x){
    r <- predict(m, x, probability = TRUE)
    prob <- attr(r, "probabilities")
    return(prob)
  }
  
  #Training from a set of distances with 1-NN as base classifier
  set.seed(3)
  m.selft1 <- selfTraining(x = dtrain, y = ytrain, x.inst = FALSE, 
                           learner = oneNN, 
                           pred = "predict", 
                           pred.pars = list(type = "prob", distance.weighting = "none"))

  #Training with learning based on nearest neighbor rule and cut edges
  set.seed(3)
  m.snnrce <- snnrce(x = xtrain, y = ytrain, dist = "Euclidean")
  
  ###Training with SElf-TRaining with EDiting. SETRED uses an amending scheme to avoid the introduction of noisy examples into the enlarged labeled set. 
  #For each iteration, the mislabeled examples are identified using the local information provided by the neighborhood graph.
  set.seed(3)
  m.setred <- setred(x = dtrain, y = ytrain, x.inst=FALSE, 
                     learner = ssc::oneNN, pred = "predict",
                     pred.pars = list(type = "prob", distance.weighting = "none"))

####Prediction###3
  m = list(m.selft1,  m.setred,  m.snnrce)
  names(m) = c("m.selft1", "m.setred",  "m.snnrce")
  rlist::list.save(m, file = paste0(output_tbl_dir, 'models_selft_setred_snnrce_1NN.RData'))
  
  j = 0
  i_label_data = as.data.frame(yitest)
  colnames(i_label_data) = "TRUTH"
  t_label_data = as.data.frame(yttest)
  colnames(t_label_data) = "TRUTH"
  
  source("/rgs01/home/clusterHome/qtran/Semisupervised_Learning/Rscripts/predict_setred_prob.R")
  for (i in m){
    trainer = attr(i, 'class')
    learner = attr(i$model, 'class')
    if (is.null(learner)){
      learner = j + 1
    }
    
    ###perform prediction
    ipred = get_inductive_pred(i, test_x=xitest, test_y=yitest, train_x=xtrain)
    transpred = get_trans_pred(i, train_x = xtrain, trans_y=yttest, tra.na.idx)
    ###create confusion matrices
    iconM = createConfusionMatrix(ipred, yitest, mode = "everything")
    transconM = createConfusionMatrix(transpred, yttest, mode="everything")
    
    write.csv(iconM$byClass, file=paste0(output_tbl_dir,"/Inductive_metrics_", trainer, "_", learner, ".csv"))
    write.csv(transconM$byClass, file=paste0(output_tbl_dir, "/Transductive_metrics_", trainer, "_", learner, ".csv"))
    
    ####plot confusion matrices
    iPT = create_plotTable(iconM)
    plot_CM(iPT, filename = paste0(output_fig_dir, "/Inductive_", trainer, "_", learner, ".pdf"))
    pPT = create_plotTable(transconM)
    plot_CM(pPT, filename = paste0(output_fig_dir, "/Transductive_", trainer, "_", learner, ".pdf"))
    
    ###accuracy
    acc = rbind(acc, c(trainer, learner, s, iconM$overall[1], transconM$overall[1]))
    write.csv(acc, file =paste0(output_tbl_dir, "/ACC_ALL.csv"))
    
    ###get predicted labels
    j = j + 1
    i_label_data = cbind(i_label_data, createLabelData(rownames(xitest), ipred)$class)
    colnames(i_label_data)[j+1] = paste0(trainer, "_", learner) 
    t_label_data = cbind(t_label_data, createLabelData(rownames(xttest), transpred)$class)
    colnames(t_label_data)[j+1] = paste0(trainer, "_", learner) 
    
    
    ###get predicted probabilities
    print(paste("j=",j))
    #
    if(trainer =="selfTraining" || trainer =="snnrce" || trainer =="setred"){
      ipred_score = predProb.OneNN(i, ditest[, i$instances.index])
      rownames(ipred_score) = rownames(ditest)
      tpred_score = predProb.OneNN(i, dttest[, i$instances.index])
      rownames(tpred_score) = rownames(dttest)
      write.csv(ipred_score, file=paste0(output_tbl_dir,"/Inductive_scores_", trainer, "_", learner, ".csv"), row.names=TRUE)
      write.csv(tpred_score, file = paste0(output_tbl_dir,"/Transductive_scores_", trainer, "_", learner, ".csv"), row.names=TRUE)
      
      ioutscores = create_class_score_output(ipred, ipred_score, yitest)
      toutscores = create_class_score_output(transpred, tpred_score, yttest)
      write.csv(ioutscores, file=paste0(output_tbl_dir,"/Inductive_MaxScores_", trainer, "_", learner, ".csv"), row.names=TRUE)
      write.csv(toutscores, file = paste0(output_tbl_dir,"/Transductive_MaxScores_", trainer, "_", learner, ".csv"), row.names=TRUE)
    }
  }
  #write.csv(acc, file =paste0(output_tbl_dir, "/ACC_SELFT.csv"))
  write.csv(i_label_data, file=paste0(output_tbl_dir, "/Inductive_predicted_SELFT_SNNRCE_SETRED_labels.csv"), row.names=TRUE)
  write.csv(t_label_data, file=paste0(output_tbl_dir, "/Transductive_predicted_SELFT_SNNRCE_SETRED_labels.csv"), row.names=TRUE)
  
  print(paste0("Complete run for seed = ", s))
}
