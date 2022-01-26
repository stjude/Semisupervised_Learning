library(randomForest)
#library(mlbench)
library(MLmetrics)
library(caret)
library(e1071)
library(doMC)
registerDoMC(cores=7)

#library(doParallel)
#cl <- detectCores() - 1  
#registerDoParallel(cores=cl)

setwd("/research/rgs01/home/clusterHome/qtran/Semisupervised_Learning/")
load("./processed_data/class_results/seed2_alldata_sets.RData")

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

data_train = data.frame(xtrain_nonNA)
data_train$meth_label = ytrain_nonNA
data_train$meth_label = as.factor(data_train$meth_label)
# Define the control
control <- trainControl(method = "repeatedcv",number = 3,search = "grid", repeats=10, classProbs = TRUE,verboseIter = TRUE,
                        returnResamp = 'none', summaryFunction  = multiClassSummary, allowParallel = TRUE, na.omit, sampling = "down",
                        returnData = FALSE, trim=TRUE)

#cl<-makePSOCKcluster(6)
#registerDoParallel(cl)
seed = 1234
set.seed(seed)
temp = dplyr::count(data_train, meth_label) 
nmin = min(temp$n)
rm(temp)
rm(data_train)
#rf1 = randomForest(meth_label ~., data_train, mtry =10, maxnodes = 200, ntree=2000, nodesize=3,
#                   sampsize = rep(nmin, 91), 
#                   strata = data_train$meth_label)
tuneGrid <- expand.grid(.mtry = 10)
rf1 <- train(x = xtrain_nonNA,
             y = make.names(ytrain_nonNA),
             method = "rf",
             metric = "Accuracy",
             tuneGrid = tuneGrid,
             trControl = control,
             #importance = TRUE,
             nodesize = 3,
             maxnodes = 200,
             ntree=2000,
             strata = ytrain_nonNA, 
             sampsize = rep(nmin, 91))

predictions = as.numeric(predict(rf1, xitest))
library(pROC)
rf.roc = multiclass.roc(as.factor(yitest), predictions)
auc(rf.roc)

save(rf1, file = "./processed_data/class_results/rf1_seed2_class.RData" )
tClasses1 <- predict(rf1, newdata = xitest)
yitest = gsub(", ", "..", yitest)
yitest = gsub(" ", ".", yitest)
predict1 = confusionMatrix(tClasses1, as.factor(yitest))
write.table(tClasses1, file = "./processed_data/class_results/rf1_predicted_label_seed2.txt")
save(predict1, file = "./processed_data/class_results/rf1_seed2_Ind_CM.RData")

########### DRAW confusion matrix for
#load("./processed_data/class_results/seed20_rf1_confusionMatrix.RData")
library(cvms)
make_plot_table = function(iLabel, predictLabel){
  plt <- data.frame(
    "target" = iLabel,
    "prediction" = predictLabel,
    stringsAsFactors = FALSE
  )
  #print(plt)
  colnames(plt) = c("Reference", "Prediction")
  plt$Reference = gsub("\\.\\.", ", ", plt$Reference)
  plt$Reference = gsub("\\.", " ", plt$Reference)
  plt$Prediction = gsub("\\.\\.", ", ", plt$Prediction)
  plt$Prediction = gsub("\\.", " ", plt$Prediction)
  #plt$Prediction <- factor(plt$Prediction, levels=rev(levels(plt$Prediction)))
  return (plt)
}

get_perm_metric = function(iLabel, predictLabel){
  colnames(predictLabel) = gsub("\\.\\.", ", ", colnames(predictLabel))
  colnames(predictLabel) = gsub("\\.", " ", colnames(predictLabel))
  plt = cbind(predictLabel, as.data.frame(iLabel))
  plt$iLabel = gsub("\\.\\.", ", ", plt$iLabel)
  plt$iLabel = gsub("\\.", " ", plt$iLabel)
  
  eval = cvms::evaluate(
    data=plt,
    target_col = "iLabel",
    prediction_cols = colnames(predictLabel),
    type="multinomial"
  )
  return (eval)
}

#plotT = make_plot_table(iLabel = yitest, predictLabel = tClasses1)
tClasses1 <- predict(rf1, newdata = xitest, type='prob')
eval1 = get_perm_metric(yitest, tClasses1)
save(eval1, file = "./processed_data/class_results/rf1_seed2_itest_eval.RData")
color_key = read.csv("/research/rgs01/home/clusterHome/qtran/Capper_reference_set/color.key.allRef.csv")
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
plot_cm(eval1, filename = "./figures/class_results/rf1_seed2_ind_testing_CM.pdf", color_key)
#rs <- rf.roc[['rocs']]
#plot.roc(rs[[1]])
#sapply(2:length(rs),function(i) lines.roc(rs[[i]],col=i))
####best maxnodes = 200 ####
##best_mtry = 10
###best_ntree = 2000 yielded mean_balanced_ACC= 71.2%, AUC = 99.79%, logLoss=1.995
set.seed(1234)
best_mtry = 10
tuneGrid <- expand.grid(.mtry = best_mtry)

library(dplyr)
#########ADD more data to training. Add the data from transductive testing to the training set######
#####read predicted label from transductive test set after calibration
yttest = read.csv("./processed_data/class_results/Calibrated_Trans_scores_long.csv", header=TRUE)
####select the score from seed20 which yielded the best SETRED-SVM
yttest_seed2 = yttest[yttest$seed == "seed2_all", ]
###get the label with max scores
#aggregate(pred_score ~ Sample, data=yttest_seed20, max)
labelseed2 = merge(aggregate(pred_score ~ Sample, data=yttest_seed2, max), yttest_seed2, all.x=T)

####combine with the xtrain to make a new train data set with predicted transductive labels
comb_data = as.data.frame(cbind(xtrain, ytrain))
comb_data$ytrain[which(is.na(comb_data$ytrain))] = 
  labelseed2$pred_label[match(rownames(comb_data)[which(is.na(comb_data$ytrain))] ,labelseed2$Sample)]

temp2 = dplyr::count(comb_data, ytrain) 
nmin2 = min(temp2$n)
rm(temp2)
comb_data$ytrain = as.factor(comb_data$ytrain)

rf2 <- train(make.names(ytrain) ~.,
             data = comb_data,
             method = "rf",
             metric = "Accuracy",
             tuneGrid = tuneGrid,
             trControl = control,
             #importance = TRUE,
             nodesize = 3,
             maxnodes = 200,
             ntree = 2000,
             strata = comb_data$ytrain,
             sampsize = rep(nmin2, 91))

save(rf2, file = "./processed_data/class_results/rf2_seed2_add35Train_class.RData")

#rf2 = randomForest(ytrain ~., comb_data, mtry =10, maxnodes = 200, ntree=2000, nodesize=3,
#                   sampsize = rep(nmin2, 91), 
#                   strata = comb_data$ytrain)
####################
tClasses2 <- predict(rf2, newdata = xitest)
#yitest = gsub(", ", "..", yitest)
#yitest = gsub(" ", ".", yitest)
predict2 = confusionMatrix(tClasses2, as.factor(yitest))
write.table(tClasses2, file = "./processed_data/class_results/rf2_predicted_label_seed2.txt")
save(predict2, file = "./processed_data/class_results/rf2_seed2_ind_CM.RData")

#########DRAW#######
tClasses2 <- predict(rf2, newdata = xitest, type='prob')
eval2 = get_perm_metric(yitest, tClasses2)

save(eval2, file = "./processed_data/class_results/rf2_seed2_itest_eval.RData")

color_key = read.csv("/research/rgs01/home/clusterHome/qtran/Capper_reference_set/color.key.allRef.csv")
#color_key = read.csv("/Volumes/qtran/Capper_reference_set/color.key.allRef.csv")

plot_cm(eval2, filename = "./figures/class_results/rf2_seed2_ind_testing_CM.pdf", color_key)

############