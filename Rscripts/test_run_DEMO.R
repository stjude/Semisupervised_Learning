#library(data.table)

load("/research/rgs01/home/clusterHome/qtran/SJ_validation_set/label_data_x.RData")
load("/research/rgs01/home/clusterHome/qtran/SJ_validation_set/label_data_y.RData")



source("/research/rgs01/home/clusterHome/qtran/SJ_validation_set/supporting_functions.R")
###Use 70% of the Capper Reference for training
##first get the index of the labeled data
acc = data.frame("Train_function" = "", "Learner"="", "Seed" = "", "Ind_ACC"="", "Trans_ACC" = "")
main_dir = "/research/rgs01/home/clusterHome/qtran/SJ_validation_set"
for (s in c(5, 50, 100, 150, 200)){
  print(paste0("Start with seed = ", s))
  
  sub_dir = paste0("seed", s)
  output_tbl_dir <- file.path(main_dir, "processed_data", sub_dir)
  output_fig_dir <- file.path(main_dir, "figures", sub_dir)
  if (!dir.exists(output_tbl_dir) && ! dir.exists(output_fig_dir)){
    dir.create(output_tbl_dir)
    dir.create(output_fig_dir)
  } 
  else {
    print("Dir already exists! Save to the existing dir")
  
    tra.idx = get_fraction_idx(label_data_x, label_data_y, seed=s, fraction = 0.7)
    
    #tra.idx = get_sample_n_rows_idx(label_data_x, label_data_y, n=nums$min_class, replace=FALSE)
    ####make train, test data sets#######
    xtrain = NULL
    ytrain = NULL
    xtrain <- label_data_x[tra.idx,] # training instances
    ytrain <- label_data_y[tra.idx] # classes of training instances
    
    ###Use 50% of train instances as unlabeled set
    tra.na.idx <- get_fraction_idx(xtrain, ytrain, seed=s, fraction = 0.5)
    ytrain[tra.na.idx] <- NA
    
    #Use the other 30% of instances for inductive test
    tst.idx <- setdiff(1:length(label_data_y), tra.idx)
    xitest <- label_data_x[tst.idx,] # test instances
    yitest <- label_data_y[tst.idx] # classes of instances in xitest
    
    # Use the unlabeled examples for transductive test
    xttest <- label_data_x[tra.idx[tra.na.idx],] # transductive test instances 
    yttest <- label_data_y[tra.idx[tra.na.idx]] # classes of instances in xttest
    
    
    #############base line supervised learning classifier SVM##########
    library(caret)
    library(ssc)
    library(kernlab)
    
    ################ computing distance and kernel matrices
    dtrain <- proxy::dist(x = xtrain, method = "euclidean", by_rows = TRUE) 
    ditest <- as.matrix(proxy::dist(x = xitest, y = xtrain, method = "euclidean", by_rows = TRUE))
    
    ###Set up learner parameters 
    learner <- e1071::svm
    learner.pars <- list(type = "C-classification", kernel="radial", 
                         probability = TRUE, scale = TRUE)
    pred <- function(m, x){
      r <- predict(m, x, probability = TRUE)
      prob <- attr(r, "probabilities")
      return(prob)
    }
    
    #set.seed(3)
    #m.snnrce <- snnrce(x = xtrain, y = ytrain, dist = "Euclidean")
    library(C50)
    set.seed(3)
    #m.demo <- democratic(x = xtrain, y = ytrain, learners = list(knn3, learner, C5.0),
    #                    learners.pars = list(list(k=1), learner.pars, NULL), 
    #                     preds = list(predict, pred, predict),
    #                     preds.pars=list(NULL, NULL, list(type = "prob"))
    #)
    m.demo <- democratic(x = xtrain, y = ytrain, learners = list(knn3, ksvm, C5.0),
                         learners.pars = list(list(k=1), list(prob.model = TRUE), NULL), 
                         preds = list(predict, predict, predict), 
                         preds.pars =list(NULL, list(type = "probabilities"), list(type = "prob"))
    )
    
    save(m.demo, file = paste0(output_tbl_dir,"/m_demo.RData"))
    #source("/research/rgs01/home/clusterHome/qtran/SJ_validation_set/predictOneNN.R")
    #base_dir = "/research/rgs01/home/clusterHome/qtran/SJ_validation_set/"
    #m = list(m.selft1, m.selft2, m.selft3, m.setred, m.setred2, m.setred3, m.snnrce, m.trit, m.trit2, m.trit3, m.demo)
    m = list(m.demo)
    j = 0
    i_label_data = as.data.frame(yitest)
    colnames(i_label_data) = "TRUTH"
    t_label_data = as.data.frame(yttest)
    colnames(t_label_data) = "TRUTH"
    acc = data.frame("Train_function" = "", "Learner"="", "Ind_ACC"="", "Trans_ACC" = "")
    
    for (i in m){
      trainer = attr(i, 'class')
      learner = attr(i$model, 'class')
      ###perform prediction
      ipred = get_inductive_pred(i, test_x=xitest, test_y=yitest, train_x=xtrain)
      transpred = get_trans_pred(i, train_x = xtrain, trans_y=yttest, tra.na.idx)
      
      ###create confusion matrices
      iconM = createConfusionMatrix(ipred, yitest)
      transconM = createConfusionMatrix(transpred, yttest)
      write.csv(iconM$byClass, file=paste0(output_tbl_dir,"/Inductive_metrics_", trainer, "_", learner, ".csv"))
      write.csv(transconM$byClass, file=paste0(output_tbl_dir, "/Transductive_metrics_", trainer, "_", learner, ".csv"))
      
      ####plot confusiong matrices
      iPT = create_plotTable(iconM)
      plot_CM(iPT, filename = paste0(output_fig_dir, "/Inductive_", trainer, "_", learner, ".pdf"))
      pPT = create_plotTable(transconM)
      plot_CM(pPT, filename = paste0(output_fig_dir, "/Transductive_", trainer, "_", learner, ".pdf"))
      
      ###accuracy
      acc = rbind(acc, c(trainer, learner, iconM$overall[1], transconM$overall[1]))
      write.csv(acc, file =paste0(output_tbl_dir, "/ACC_DEMO.csv"))
      
      ###get predicted labels
      j = j + 1
      i_label_data = cbind(i_label_data, createLabelData(rownames(xitest), ipred)$class)
      colnames(i_label_data)[j+1] = paste0(trainer, "_", learner) 
      t_label_data = cbind(t_label_data, createLabelData(rownames(xttest), transpred)$class)
      colnames(t_label_data)[j+1] = paste0(trainer, "_", learner) 
      
      #write.csv(i_label_data, file=paste0(base_dir, "processed_data/Inductive_predicted_DEMO_labels.csv"))
      #write.csv(t_label_data, file=paste0(base_dir, "processed_data/Transductive_predicted_DEMO_labels.csv"))
      
      ###get predicted probabilities
      print(paste("j=",j))
    }
    
    write.csv(acc, file =paste0(output_tbl_dir, "/ACC_DEMO.csv"))
    write.csv(i_label_data, file=paste0(output_tbl_dir, "/Inductive_predicted_DEMO_labels.csv"))
    write.csv(t_label_data, file=paste0(output_tbl_dir, "/Transductive_predicted_DEMO_labels.csv"))
    print(paste0("Complete run for seed = ", s))
    
  }
  
}