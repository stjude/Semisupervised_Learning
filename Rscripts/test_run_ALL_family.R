load("/research/rgs01/home/clusterHome/qtran/Semisupervised_Learning/label_data_x.RData")
load("/research/rgs01/home/clusterHome/qtran/Semisupervised_Learning/family_label_data_y.RData")
source("/research/rgs01/home/clusterHome/qtran/Semisupervised_Learning/supporting_functions.R")
source("/research/rgs01/home/clusterHome/qtran/Semisupervised_Learning/predictOneNN.R")
###Use 70% of the Capper Reference for training
##first get the index of the labeled data
acc = data.frame("Train_function" = "", "Learner"="", "Seed" = "", "Ind_ACC"="", "Trans_ACC" = "")
main_dir = "/research/rgs01/home/clusterHome/qtran/Semisupervised_Learning"

create_class_score_output = function(pred_label, pred_score, truth_label){
  output = data.frame(pred_label)
  output$Sample = names(truth_label)
  output$pred_score = apply(pred_score, 1, max)
  output$truth_label = truth_label
  
  return(output)
}

for (s in c(1, 2, 20, 40, 80, 160, 320)){
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
  
  tra.idx = get_fraction_idx(label_data_x, label_data_y, seed=s, fraction = 0.7)
  
  #tra.idx = get_sample_n_rows_idx(label_data_x, label_data_y, n=nums$min_class, replace=FALSE)
  ####make train, test data sets#######
  xtrain = NULL
  ytrain = NULL
  xtrain <- label_data_x[tra.idx,] # training instances
  ytrain <- label_data_y[tra.idx] # classes of training instances
  
  ###Use 50% of train instances as unlabeled set
  tra.na.idx <- get_fraction_idx(xtrain, ytrain, seed=25, fraction = 0.5)
  ytrain[tra.na.idx] <- NA
  
  #Use the other 30% of instances for inductive test
  tst.idx <- setdiff(1:length(label_data_y), tra.idx)
  xitest <- label_data_x[tst.idx,] # test instances
  yitest <- label_data_y[tst.idx] # classes of instances in xitest
  
  # Use the unlabeled examples for transductive test
  xttest <- label_data_x[tra.idx[tra.na.idx],] # transductive test instances 
  yttest <- label_data_y[tra.idx[tra.na.idx]] # classes of instances in xttest
  
  d = list(xtrain, ytrain, xitest, yitest, xttest, yttest, tra.idx, tra.na.idx)
  names(d) = c("xtrain", "ytrain", "xitest", "yitest", "xttest", "yttest", "tra.idx", "tra.na.idx")
  rlist::list.save(d, file = paste0(output_tbl_dir, 'data_sets.RData'))

  #############base line supervised learning classifier SVM##########
  library(caret)
  library(ssc)
  library(kernlab)
  
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
  set.seed(3)
  m.selft2 <- selfTraining(x = xtrain, y = ytrain, x.inst=TRUE, 
                           learner = learner, 
                           learner.pars = learner.pars, 
                           pred = pred) 
  #Self-Training from a set of instances with decision tree learning method
  set.seed(3)
  m.selft3<- selfTraining(x = xtrain, y = ytrain, learner = C50::C5.0, pred = "predict", learner.pars = list(prob.model = TRUE), 
                          pred.pars = list(type = "prob"))
  
  #Self-Training from a set of instances using Naive Bayes learning method
  #m.selft4 <- selfTraining(x = dtrain, y = ytrain, x.inst=FALSE, 
  #                         learner = function(x, y) e1071::naiveBayes(x, y), 
  #                         pred = "predict", 
  #                         pred.pars = list(type = "raw"))
  
  
  #Training with learning based on nearest neighbor rule and cut edges
  set.seed(3)
  m.snnrce <- snnrce(x = xtrain, y = ytrain, dist = "Euclidean")
  
  ###Training with SElf-TRaining with EDiting. SETRED uses an amending scheme to avoid the introduction of noisy examples into the enlarged labeled set. 
  #For each iteration, the mislabeled examples are identified using the local information provided by the neighborhood graph.
  set.seed(3)
  m.setred <- setred(x = dtrain, y = ytrain, x.inst=FALSE, 
                     learner = ssc::oneNN, pred = "predict",
                     pred.pars = list(distance.weighting = "none"))
  
  #Self-training with EDiting using SVM learner
  set.seed(3)
  m.setred2 <- setred(x = xtrain, y = ytrain, dist = "euclidean", 
                      learner = learner, 
                      learner.pars = learner.pars, 
                      pred = pred)
  set.seed(3)
  m.setred3 <- setred(x = xtrain, y = ytrain, dist="euclidean",
                      learner = C50::C5.0, 
                      pred = "predict",  
                      pred.pars = list(type = "prob"))
  
  set.seed(3)
  m.trit <- triTraining(x = xtrain, y = ytrain, x.inst = TRUE, learner = knn3,  pred = "predict", 
                        learner.pars = list(k = 1),
                        pred.pars = list(type = "prob"))
  set.seed(3)
  m.trit2 <- triTraining(x = xtrain, y=ytrain,
                         x.inst=TRUE, learner = learner,
                         learner.pars = learner.pars,
                         pred = pred)
  set.seed(3)
  m.trit3 <- triTraining(x = xtrain, y=ytrain,
                         x.inst=TRUE, learner = C50::C5.0,
                         pred.pars = list(type = "prob"), 
                         pred = "predict")
  
  library(C50)
  set.seed(3)

  m.demo <- democratic(x = xtrain, y = ytrain, learners = list(knn3, ksvm, C5.0),
                       learners.pars = list(list(k=1), list(prob.model = TRUE), NULL), 
                       preds = list(predict, predict, predict), 
                       preds.pars =list(NULL, list(type = "probabilities"), list(type = "prob"))
  )
  
####Prediction###3
  m = list(m.selft1, m.selft2, m.selft3, m.setred, m.setred2, m.setred3, m.snnrce, m.trit, m.trit2, m.trit3, m.demo)
  names(m) = c("m.selft1", "m.self2", "m.self3", "m.setred", "m.setred2", "m.setred3", "m.snnrce", "m.trit", "m.trit2", "m.trit3", "m.demo")
  rlist::list.save(m, file = paste0(output_tbl_dir, 'modelsALL.RData'))
  
  j = 0
  i_label_data = as.data.frame(yitest)
  colnames(i_label_data) = "TRUTH"
  t_label_data = as.data.frame(yttest)
  colnames(t_label_data) = "TRUTH"
  
  source("/rgs01/home/clusterHome/qtran/Semisupervised_Learning/predict_setred_prob.R")
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
    if(trainer=="setred" & learner == "svm"){
      ipred_score <- predict.setred.prob (i, xitest, type="prob")
      tpred_score <- predict.setred.prob (i, xttest, type="prob")
      write.csv(ipred_score, file=paste0(output_tbl_dir,"/Inductive_scores_", trainer, "_", learner, ".csv"))
      write.csv(tpred_score, file = paste0(output_tbl_dir,"/Transductive_scores_", trainer, "_", learner, ".csv"))
      
      ioutscores = create_class_score_output(ipred, ipred_score, yitest)
      toutscores = create_class_score_output(transpred, tpred_score, yttest)
      write.csv(ioutscores, file=paste0(output_tbl_dir,"/Inductive_MaxScores_", trainer, "_", learner, ".csv"))
      write.csv(toutscores, file = paste0(output_tbl_dir,"/Transductive_MaxScores_", trainer, "_", learner, ".csv"))
    }
    if(trainer =="selfTraining" & learner == "OneNN"){
      ipred_score = predProb.OneNN(i, ditest[, i$instances.index])
      tpred_score = predProb.OneNN(i, dttest[, i$instances.index])
      write.csv(ipred_score, file=paste0(output_tbl_dir,"/Inductive_scores_", trainer, "_", learner, ".csv"))
      write.csv(tpred_score, file = paste0(output_tbl_dir,"/Transductive_scores_", trainer, "_", learner, ".csv"))
      
      ioutscores = create_class_score_output(ipred, ipred_score, yitest)
      toutscores = create_class_score_output(transpred, tpred_score, yttest)
      write.csv(ioutscores, file=paste0(output_tbl_dir,"/Inductive_MaxScores_", trainer, "_", learner, ".csv"))
      write.csv(toutscores, file = paste0(output_tbl_dir,"/Transductive_MaxScores_", trainer, "_", learner, ".csv"))
    }
  }
  #write.csv(acc, file =paste0(output_tbl_dir, "/ACC_SELFT.csv"))
  write.csv(i_label_data, file=paste0(output_tbl_dir, "/Inductive_predicted_ALL_labels.csv"))
  write.csv(t_label_data, file=paste0(output_tbl_dir, "/Transductive_predicted_ALL_labels.csv"))
  
  print(paste0("Complete run for seed = ", s))
}
