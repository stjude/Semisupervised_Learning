library(randomForest)
#library(mlbench)
#library(MLmetrics)
library(caret)
library(cvms)
library(pROC)
#library(e1071)
###for Linux
library(doMC)
registerDoMC(cores=18)
###for MAC
#library(doParallel)
#cl <- makeCluster(7,"PSOCK")
#registerDoParallel(cl)

setwd("/research/rgs01/home/clusterHome/qtran/Semisupervised_Learning/")
load("./processed_data/class_results/seed160_alldata_sets.RData")
color_key = read.csv("/research/rgs01/home/clusterHome/qtran/Capper_reference_set/color.key.allRef.csv")

xtrain = x$xtrain
ytrain = x$ytrain
xitest = x$xitest
yitest = x$yitest
xttest = x$xttest
yttest = x$yttest
tra.na.idx = x$tra.na.idx ###indices of NA samples from the training data set of the semisupervised classifier
rm(x)

keep.idx <- setdiff(1:length(ytrain), tra.na.idx)
xtrain_nonNA = data.frame(xtrain[keep.idx,])
ytrain_nonNA = ytrain[keep.idx]
ytrain_nonNA = as.factor(ytrain_nonNA)
ytrain_nonNA = make.names(ytrain_nonNA)

#yitest = gsub(", ", "..", yitest)
#yitest = gsub(" ", ".", yitest)

####best maxnodes = 200 ####
##best_mtry = 10
###best_ntree = 2000 yielded mean_balanced_ACC= 71.2%, AUC = 99.79%, logLoss=1.995
set.seed(1234)
best_mtry = 10
tuneGrid <- expand.grid(.mtry = best_mtry)

#########ADD more data to training. Add the data from transductive testing to the training set######
#####read predicted label from transductive test set after calibration
yttest = read.csv("./processed_data/class_results/Calibrated_Trans_scores_long.csv", header=TRUE)
####select the score from seed800 which yielded the best SETRED-SVM
yttest_seed = yttest[yttest$seed == "seed160_all", ]
###get the label with max scores
#aggregate(pred_score ~ Sample, data=yttest_seed800, max)
labelseed = merge(aggregate(pred_score ~ Sample, data=yttest_seed, max), yttest_seed, all.x=T)

####combine with the xtrain to make a new train data set with predicted transductive labels
comb_data = as.data.frame(cbind(xtrain, ytrain))
comb_data$ytrain[which(is.na(comb_data$ytrain))] = 
  labelseed$pred_label[match(rownames(comb_data)[which(is.na(comb_data$ytrain))] ,labelseed$Sample)]

comb_data$ytrain = trimws(comb_data$ytrain)
comb_data$ytrain = as.factor(comb_data$ytrain)
gc()
####
set.seed(37596)

folds <- createMultiFolds(comb_data$ytrain, k = 3, times=5)
#lapply(folds, function(x, y) return(y[x,]), y = comb_data)

#ctrl <- trainControl(method = "repeatedcv", number = 3, repeats = 5,
#                       index = folds, trim=TRUE, classProbs = TRUE,verboseIter = TRUE,
#                       returnResamp = 'none', summaryFunction  = multiClassSummary, 
#                       allowParallel = TRUE, na.omit, sampling = "down")

bfreq = table(comb_data$ytrain[folds[[6]]])


####


get_perm_metric = function(iLabel, predictLabel){
  #colnames(predictLabel) = gsub("\\.\\.", ", ", colnames(predictLabel))
  #colnames(predictLabel) = gsub("\\.", " ", colnames(predictLabel))
  plt = cbind(predictLabel, as.data.frame(iLabel))
  plt$predictLabel = as.character(plt$predictLabel)
  plt$iLabel = as.character(plt$iLabel)
  print(head(plt))
  #plt$iLabel = gsub("\\.\\.", ", ", plt$iLabel)
  #plt$iLabel = gsub("\\.", " ", plt$iLabel)
  
  eval = cvms::evaluate(
    data=plt,
    target_col = "iLabel",
    prediction_cols = "predictLabel",
    type="multinomial"
  )
  return (eval)
}
make_plot_table = function(iLabel, predictLabel){
  plt <- data.frame(
    "target" = iLabel,
    "prediction" = predictLabel,
    stringsAsFactors = FALSE
  )
  #print(plt)
  colnames(plt) = c("Reference", "Prediction")
  #plt$Reference = gsub("\\.\\.", ", ", plt$Reference)
  #plt$Reference = gsub("\\.", " ", plt$Reference)
  #plt$Prediction = gsub("\\.\\.", ", ", plt$Prediction)
  #plt$Prediction = gsub("\\.", " ", plt$Prediction)
  #plt$Prediction <- factor(plt$Prediction, levels=rev(levels(plt$Prediction)))
  return (plt)
}

bacc=  list()
for(i in names(folds)){
  trainIndex = folds[[i]]
  length(folds[[i]])
  train <- comb_data[trainIndex, ]
  test <- comb_data[-trainIndex, ]
  b = as.factor(train$ytrain)
  bfreq=table(b)
  rf2 = foreach(ntree=rep(100,20), .combine=combine, .packages='randomForest') %dopar%
    randomForest(train[, 1:ncol(train)-1], y=b, 
                 ntree=ntree,do.trace=FALSE, mtry=10, maxnodes = 200, nodesize=3, keep.forest = TRUE,
                 strata=b,sampsize=rep(min(bfreq),dim(table(b))))
  predL = predict(rf2, test[, 1:ncol(test)-1])
  eval2 = get_perm_metric(test$ytrain, predL)
  bacc[[i]] = eval2
}

gc()
saveRDS(bacc, file ="./processed_data/class_results/seed160_all/rf2_acc.rds")
saveRDS(rf2, file = "./processed_data/class_results/rf2_seed160_add35Train_class.rds")


########### DRAW confusion matrix for
#load("./processed_data/class_results/seed800_rf1_confusionMatrix.RData")
#plotT = make_plot_table(iLabel = yitest, predictLabel = tClasses1)

plot_cm = function(eval, filename, col){
  g = plot_confusion_matrix(eval, add_counts = FALSE, add_row_percentages = FALSE, 
                            palette = "Reds",tile_border_color = "gray",
                            add_col_percentages = FALSE, add_arrows = FALSE, add_normalized = FALSE,
                            add_zero_shading = TRUE, darkness = 1,
                            place_x_axis_above = FALSE, rotate_y_text = FALSE)+
    ggplot2::labs(x = "Reference", y = "Prediction")+
    ggplot2::theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1, size=7, color = col$color),
                   axis.text.y = element_text(vjust = 0.5, hjust=1, size=7, color = col$color), 
                   legend.position = "bottom")
  
  ggsave(g, file = filename)
}
####################
tClasses2 <- predict(rf2, newdata = xitest)

predict2 = confusionMatrix(tClasses2, as.factor(yitest))

write.table(tClasses2, file = "./processed_data/class_results/rf2_predicted_label_seed160.txt")
save(predict2, file = "./processed_data/class_results/rf2_seed160_ind_CM.RData")

#########DRAW#######
tClasses2 <- predict(rf2, newdata = xitest, type='prob')

eval2 = get_perm_metric(yitest, tClasses2)

save(eval2, file = "./processed_data/class_results/rf2_seed160_itest_eval.RData")

#color_key = read.csv("/Volumes/qtran/Capper_reference_set/color.key.allRef.csv")

plot_cm(eval2, filename = "./figures/class_results/rf2_seed160_ind_testing_CM.pdf", color_key)

############