library(randomForest)
library(mlbench)
library(caret)
library(e1071)
library(doParallel)
cl <- detectCores() - 1  
registerDoParallel(cores=cl)

setwd("/research/rgs01/home/clusterHome/qtran/Semisupervised_Learning/")
load("./processed_data/class_results/seed20_alldata_sets.RData")

xtrain = x$xtrain
ytrain = x$ytrain
xitest = x$xitest
yitest = x$yitest
xttest = x$xttest
yttest = x$yttest
tra.na.idx = x$tra.na.idx ###indices of NA samples from the training data set of the semisupervised classifier

keep.idx <- setdiff(1:length(ytrain), tra.na.idx)
xtrain_nonNA = data.frame(xtrain[keep.idx,])
ytrain_nonNA = ytrain[keep.idx]


data_train = data.frame(xtrain_nonNA)
data_train$meth_label = ytrain_nonNA

# Define the control
trControl <- trainControl(method = "repeatedcv",number = 3,search = "grid", repeats=10, classProbs = TRUE, 
                          summaryFunction  = multiClassSummary, allowParallel = TRUE)

cl<-makePSOCKcluster(6)
registerDoParallel(cl)
seed = 1234
set.seed(seed)
mtry = sqrt(ncol(data_train))
tuneGrid <- expand.grid(.mtry = 100)
metric = "ROC"
# Run the model
rf_default <- train(make.names(meth_label) ~.,
                    data = data_train,
                    method = "rf",
                    metric = metric,
                    #tuneGrid = tuneGrid,
                    trControl = trControl, mc = cl)
# Print the results
print(rf_default)
seed = 1234
###Search the best mtry
set.seed(seed)
bestmtry <- tuneRF(x=as.matrix(xtrain_nonNA), y=as.factor(ytrain_nonNA), 
                   improve=1e-3, ntreeTry=1000, mc = cl, mtryStart=32, stepFactor = 1.5)
print(bestmtry)

####tuneRF returned mtry= 32, yielding min OOB error = 10.9%

start.time<-proc.time()
library(dplyr)
temp = dplyr::count(data_train, meth_label) 
nmin = min(temp$n)

tuneGrid <- expand.grid(.mtry = c(3, 5, 10))
rf_mtry <- train(make.names(meth_label) ~.,
                 data = data_train,
                 method = "rf",
                 metric = "ROC",
                 tuneGrid = tuneGrid,
                 trControl = trControl,
                 importance = TRUE,
                 nodesize = 3,
                 maxnodes = 200,
                 strata = data_train$meth_label,
                 sampsize = rep(nmin, 91), ####check this
                 ntree = 1000, tuneLength = 5,
                 mc =cl)
stop.time<-proc.time()
run.time<-stop.time -start.time
print(run.time)

print(rf_mtry)

best_mtry = rf_mtry$bestTune$mtry
max(rf_mtry$results$Accuracy)
#Search the best maxnode
store_maxnode <- list()
tuneGrid <- expand.grid(.mtry = best_mtry)
for (maxnodes in c(250, 300, 350, 400, 450)) {
  set.seed(1234)
  rf_maxnode <- train(make.names(meth_label) ~.,
                      data = data_train,
                      method = "rf",
                      metric = "ROC",
                      tuneGrid = tuneGrid,
                      trControl = trControl,
                      importance = TRUE,
                      nodesize = 3,
                      maxnodes = maxnodes,
                      strata = data_train$meth_label,
                      sampsize = rep(nmin, 91), 
                      ntree = 1000, mc =cl)
  current_iteration <- toString(maxnodes)
  store_maxnode[[current_iteration]] <- rf_maxnode
}

results_mtry <- resamples(store_maxnode)
summary(results_mtry)

####search best ntree####
store_ntree <- list()
tuneGrid <- expand.grid(.mtry = best_mtry)
for (ntrees in c(2000, 2500, 3000)) {
  set.seed(1234)
  rf_ntree <- train(make.names(meth_label) ~.,
                    data = data_train,
                    method = "rf",
                    metric = "ROC",
                    tuneGrid = tuneGrid,
                    trControl = trControl,
                    importance = TRUE,
                    nodesize = 3,
                    maxnodes = 200,
                    strata = data_train$meth_label,
                    sampsize = rep(nmin, 91), 
                    ntree = ntrees, mc =cl)
  current_iteration <- toString(ntrees)
  store_ntree[[current_iteration]] <- rf_ntree
}

results_ntree <- resamples(store_ntree)
summary(results_ntree)

####best maxnodes = 200 ####
##best_mtry = 10
###best_ntree = 2000 yielded mean_balanced_ACC= 71.2%, AUC = 99.79%, logLoss=1.995
set.seed(1234)
best_mtry = 10
tuneGrid <- expand.grid(.mtry = best_mtry)

trControl <- trainControl(method = "repeatedcv",number = 3,search = "grid", repeats=3, classProbs = TRUE, 
                          summaryFunction  = multiClassSummary, allowParallel = TRUE)
library(dplyr)

data_train$meth_label = as.factor(data_train$meth_label)
temp = dplyr::count(data_train, meth_label) 
nmin = min(temp$n)
rf_min <- train(make.names(meth_label) ~.,
                    data = data_train,
                    method = "rf",
                    metric = "ROC",
                    tuneGrid = tuneGrid,
                    trControl = trControl,
                    importance = TRUE,
                    nodesize = 3,
                    maxnodes = 200,
                    strata = data_train$meth_label,
                    sampsize = rep(nmin, 91), 
                    ntree = 2500,
                    mc =cl)

print(rf_min)

tClasses <- predict(rf_min, newdata = xitest)
yitest = gsub(", ", "..", yitest)
yitest = gsub(" ", ".", yitest)
predict1 = confusionMatrix(tClasses, as.factor(yitest))
write.table(tClasses, file = "./processed_data/class_results/rf1_predicted_label_seed20.txt")
save(predict1, file = "./processed_data/class_results/seed20_rf1_confusionMatrix.RData")

#########ADD more data to training. Add the data from transductive testing to the training set######
#####read predicted label from transductive test set after calibration
yttest = read.csv("./processed_data/class_results/Calibrated_Trans_scores_long.csv", header=TRUE)
####select the score from seed20 which yielded the best SETRED-SVM
yttest_seed20 = yttest[yttest$seed == "seed20_all", ]
###get the label with max scores
aggregate(pred_score ~ Sample, data=yttest_seed20, max)
labelseed20 = merge(aggregate(pred_score ~ Sample, data=yttest_seed20, max), yttest_seed20, all.x=T)

####combine with the xtrain to make a new train data set with predicted transductive labels
comb_data = as.data.frame(cbind(xtrain, ytrain))
comb_data$ytrain[which(is.na(comb_data$ytrain))] = 
  labelseed20$pred_label[match(rownames(comb_data)[which(is.na(comb_data$ytrain))] ,labelseed20$Sample)]

temp2 = dplyr::count(comb_data, ytrain) 
nmin2 = min(temp2$n)
comb_data$ytrain = as.factor(comb_data$ytrain)

trControl <- trainControl(method = "cv",number = 3,search = "grid", repeats=1, classProbs = TRUE, 
                          summaryFunction  = multiClassSummary, allowParallel = TRUE)
rf2 <- train(make.names(ytrain) ~.,
             data = comb_data,
             method = "rf",
             metric = "ROC",
             tuneGrid = tuneGrid,
             trControl = trControl,
             #importance = TRUE,
             nodesize = 3,
             maxnodes = 200,
             strata = comb_data$ytrain,
             sampsize = rep(nmin2, 91), 
             ntree = 2500,
             mc =cl)
####################
stopCluster(cl)


########### DRAW confusion matrix for
load("./processed_data/class_results/seed20_rf1_confusionMatrix.RData")
library(cvms)
library(yard)
plt <- as.data.frame(predict1$table)

plt <- data.frame(
  "target" = yitest,
  "prediction" = tClasses,
  stringsAsFactors = FALSE
)
colnames(plt) = c("Reference", "Prediction")
plt$Reference = gsub("\\.\\.", ", ", plt$Reference)
plt$Reference = gsub("\\.", " ", plt$Reference)
plt$Prediction = gsub("\\.\\.", ", ", plt$Prediction)
plt$Prediction = gsub("\\.", " ", plt$Prediction)

eval <- evaluate(
  data = plt,
  target_col = "Reference",
  prediction_cols = "Prediction",
  type = "multinomial"
)

eval
plt$Prediction <- factor(plt$Prediction, levels=rev(levels(plt$Prediction)))
plot_confusion_matrix(eval, add_counts = FALSE, add_row_percentages = FALSE, 
                      palette = "Reds",tile_border_color = "white",
                      add_col_percentages = FALSE, add_arrows = FALSE, add_normalized = FALSE,
                      add_zero_shading = FALSE, darkness = 1,
                      place_x_axis_above = FALSE, rotate_y_text = FALSE)+
  ggplot2::labs(x = "Reference", y = "Prediction")+
  ggplot2::theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1, size=7),
                 axis.text.y = element_text(vjust = 0.5, hjust=1, size=7), 
                 legend.position = "right")

############