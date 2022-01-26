library(data.table)
#top_beta = fread("/research/rgs01/home/clusterHome/qtran/Medulloblastoma-ORR/processed_data/top_32K_950MB_nature_all.csv", sep=",")
top_beta = fread("/research/rgs01/home/clusterHome/qtran/SJ_validation_set/processed_data/top_32K_SJ_GSE109379_GSE90496.txt")
tsne_data = fread("/research/rgs01/home/clusterHome/qtran/Medulloblastoma-ORR/processed_data/TSNE_32Kprobes_950MB_70PCs.csv", sep=",")
positive_set = read.csv("/research/rgs01/home/clusterHome/qtran/SJ_validation_set/raw_data/BT_classifier_final_validation_positiveset_20210608.csv")

top_beta = fread("/Volumes/qtran/SJ_validation_set/processed_data/top_32K_SJ_GSE109379_GSE90496.txt")
tsne_data = fread("/Volumes/qtran/Medulloblastoma-ORR/processed_data/TSNE_32Kprobes_950MB_70PCs.csv", sep=",")
positive_set = read.csv("/Volumes/qtran/SJ_validation_set/raw_data/BT_classifier_final_validation_positiveset_20210608.csv")

top_beta = as.data.frame(top_beta)
rownames(top_beta) = top_beta$V1
top_beta = top_beta[,-1]

tsne_data = tsne_data[,-1]

keep_sj_index = which(colnames(top_beta) %in% positive_set$SentrixID2)
new_beta = top_beta[1:5000, c(keep_sj_index, 951:ncol(top_beta))]

###unlabeled data
unlabel_data = new_beta[, 1:1804]
unlabel_data_x = t(unlabel_data)
unlabel_data_y = data.frame(title.class.x = rep(NA, 1804), row.names = colnames(top_beta)[1:1804])

label_samples = tsne_data$Sample[tsne_data$title.class.x != "SJ, MB"]
label_data = new_beta[, (colnames(new_beta) %in% label_samples)]
label_data_x = t(label_data)

label_data_y = tsne_data[match(rownames(label_data_x), tsne_data$Sample), c("Sample", "title.class.x")]
rownames(label_data_y) = label_data_y$Sample
label_data_y = label_data_y[ , title.class.x]
##
#@data: a data frame of beta values: rows are samples, columns are Cpgs
#@sampleName: list of a sample (ID) for the @data 
#@labels: list of the class for each sample 
#@seed: seed for random sampling
#@fraction: the proportion of data to keep
#return the index of of the rows (samples) that were randomly selected by the defined fraction
## 
library(dplyr)
get_fraction_idx = function(data, labels, seed, fraction){
  data = as.data.frame(data)
  data$class = labels
  data$Sample = rownames(data)
  ##randomly sample 80% of the samples in each class
  train_sample = data %>% group_by(class) %>% sample_frac(fraction)
  ##
  idx = match(data$Sample, train_sample$Sample)
  idx = idx[!(is.na(idx))]
  return(idx)
}
tra.idx = get_fraction_idx(label_data_x, label_data_y, seed=3, fraction = 0.8)


####make train, test data sets#######
xtrain = NULL
# Use 90% of nature data set for training
#tra.idx <- sample(x = length(label_data_y), size = ceiling(length(label_data_y) * 0.9)) 
xtrain <- label_data_x[tra.idx,] # training instances
ytrain <- label_data_y[tra.idx] # classes of training instances

#the SJ data set is unlabeled set (this can be used as transductive test later). Use the model to get the label for the unlabeled data set
xtrain = rbind(xtrain, unlabel_data_x)
ytrain = c(ytrain, rep(NA, 1804))  # combine class of unlabeled instances
tra.na.idx <- seq(length(tra.idx)+1, nrow(xtrain))

# Use the other 10% of nature data set for inductive test. This is a labeled set and hence can be checked for accuracy
tst.idx <- setdiff(1:length(label_data_y), tra.idx)
xitest <- label_data_x[tst.idx,] # test instances
yitest <- label_data_y[tst.idx] # classes of instances in xitest

#############base line supervised learning classifier SVM##########
library(caret)
library(ssc)
library(kernlab)
#labeled.idx <- which(!is.na(ytrain))# indices of the initially labeled instances 
#xilabeled <- xtrain[labeled.idx,] # labeled instances
#yilabeled <- ytrain[labeled.idx] # related classes
#svmBL <- ksvm(x = xilabeled, y = yilabeled, prob.model = TRUE, scaled = FALSE, type= 'kbb-svc', kernel = "vanilladot") # build SVM 
#p.svmBL <- predict(object = svmBL, newdata = xitest) # classify with SVM

################ computing distance and kernel matrices
dtrain <- as.matrix(proxy::dist(x = xtrain, method = "euclidean", by_rows = TRUE)) 
ditest <- as.matrix(proxy::dist(x = xitest, y = xtrain, method = "euclidean", by_rows = TRUE))

#ktrain <- as.matrix(exp(- 0.048 * dtrain^2)) 
#kitest <- as.matrix(exp(- 0.048 * ditest^2))

#Training from a set of instances with 1-NN as base classifier
#set.seed(3)
#m.selft1 <- selfTraining(x = xtrain, y = ytrain, learner = knn3, learner.pars = list(k = 1), pred = "predict")

set.seed(3)
m.selft2 <- selfTraining(x = dtrain, y = ytrain, x.inst = FALSE, learner = oneNN, pred = "predict", pred.pars = list(type = "prob"))
#Training from a set of instances with ksvm learning method
set.seed(3)
m.selft3 <- selfTraining(x = xtrain, y = ytrain, learner = ksvm, pred = "predict", learner.pars = list(prob.model = TRUE), 
                          pred.pars = list(type = "prob"))

#Training with learning based on nearest neighbor rule and cut edges
set.seed(3)
m.snnrce <- snnrce(x = xtrain, y = ytrain, dist = "Euclidean")

###Training
set.seed(3)
m.cobc <- coBC(x = xtrain, y = ytrain, N = 5, learner = ksvm, learner.pars = list(prob.model = TRUE), pred = predict,
               pred.pars = list(type = "probabilities"))
#Training from a kernel matrix using ksvm as a base classifier
#set.seed(3)
#m.selft3 <- selfTraining(x = ktrain, y = ytrain, x.inst = FALSE, learner = ksvm,
#                       learner.pars = list(kernel = "matrix", prob.model = TRUE), 
#                       pred = function(m, k) predict(m, as.kernelMatrix(k[, SVindex(m)]), type = "probabilities"))
#save(m.selft3, file ="/research/rgs01/home/clusterHome/qtran/SJ_validation_set/m.selft3.RData" )

#training with co-learning method that leverages the agreement of three independently trained models to reduce the bias of predictions on unlabeled data
set.seed(3)
m.trit <- triTraining(x = xtrain, y = ytrain, learner = ksvm, learner.pars = list(prob.model = TRUE), pred = predict, 
                      pred.pars = list(type = "probabilities"))

library(C50)
#train model with different inductive biases.Democratic co-learning first trains each model separately on the complete labelled data L.
#The models then make predictions on the unlabelled data U. 
#If a majority of models confidently agree on the label of an example, the example is added to the labelled dataset. 
#C5.0 is the decision tree using the concept of entropy for measuring purity
set.seed(3)
m.demo <- democratic(x = xtrain, y = ytrain, learners = list(knn3, ksvm, C5.0),
                     learners.pars = list(list(k=1), list(prob.model = TRUE), NULL), preds = list(predict, predict, predict),
                     preds.pars=list(NULL, list(type = "probabilities"), list(type = "prob"))
)

############Prediction##########
#load("/research/rgs01/home/clusterHome/qtran/SJ_validation_set/m.selft1.RData")
#load("/research/rgs01/home/clusterHome/qtran/SJ_validation_set/m.selft2.RData")
#load("/research/rgs01/home/clusterHome/qtran/SJ_validation_set/m.selft3.RData")

p.selft1 <- predict(m.selft1, xitest)
#p.selft2 <- predict(m.selft2, ditest[, m.selft2$instances.index])
#p.selft3 <- predict(m.selft3, as.kernelMatrix(kitest[, m.selft3$instances.index]))
source("predictOneNN.R")
p.selft2.prob <- predict.OneNN(m.selft2, ditest[, m.selft2$instances.index], type="prob")
p.selft2.class <- predict.OneNN(m.selft2, ditest[, m.selft2$instances.index], type="class")

p.snnrce <- predict(m.snnrce, xitest)
p.cobc <- predict.coBC(m.cobc, xitest, type="prob")
p.trit <- predict(m.trit, xitest)
p.demo <- predict(m.demo, xitest)

createConfusionMatrix = function(predict, actual){
  u <- union(predict, actual)
  t <- table(factor(predict, u), factor(actual, u))
  return(confusionMatrix(t))
}
createLabelData = function(sample, predict_label){
  d = data.frame(Sample = sample, class = predict_label)
  return (d)
}

p = list(p.selft1, p.selft2, p.snnrce, p.cobc, p.trit, p.demo)
acc <- sapply(X = p, FUN = function(i) {createConfusionMatrix(i,yitest)$overall[1]})

names(acc) <- c("SelfT 1-NN (beta)", "SelfT kSVM ", "SNNRCE", "Co-BC", "TriT", "Demo")
pdf("/research/rgs01/home/clusterHome/qtran/SJ_validation_set/processed_data/", height=7, width=7)
barplot(acc, beside = T, ylim = c(0.0,1), xpd = FALSE, las = 2, 
        col=rainbow(n = 6, start = 3/6, end = 4/6, alpha = 0.6) , ylab = "Accuracy")
dev.off()

save(m.selft1, file ="/research/rgs01/home/clusterHome/qtran/SJ_validation_set/m.selft1.RData")
save(m.selft2, file ="/research/rgs01/home/clusterHome/qtran/SJ_validation_set/m.selft2.RData")
#predict the unlabeled data in the training data set
p.selft1transd <- predict(m.selft1, xtrain[tra.na.idx,])
p.selft2transd <- predict(m.selft2, xtrain[tra.na.idx,])
p.snnrcetransd <- predict(m.snnrce, xtrain[tra.na.idx,])
#p.selft2.prob <- predict.OneNN(m.selft2, ditest[, m.selft2$instances.index], type="prob")
p.selft2transd.prob <- predict.OneNN(m.selft2, dtrain[tra.na.idx, m.selft2$instances.index], type="prob")
p.cobctransd <- predict(m.cobc, xtrain[tra.na.idx,])
#p.selft3transd <- predict(m.selft3, as.kernelMatrix(ktrain[tra.na.idx,m.selft3$instances.index]))
p.triTtransd <- predict(m.trit, xtrain[tra.na.idx,])
p.demotransd <- predict(m.demo, xtrain[tra.na.idx,])

d_selft1 = createLabelData(rownames(xtrain)[tra.na.idx], p.selft1transd)
d_selft2 = createLabelData(rownames(xtrain)[tra.na.idx], p.selft2transd)
d_snnrce = createLabelData(rownames(xtrain)[tra.na.idx], p.snnrcetransd)
d_cobc = createLabelData(rownames(xtrain)[tra.na.idx], p.cobctransd)
d_triT = createLabelData(rownames(xtrain)[tra.na.idx], p.triTtransd)
d_demo = createLabelData(rownames(xtrain)[tra.na.idx], p.demotransd)


write.csv(d_selft1, file="/research/rgs01/home/clusterHome/qtran/SJ_validation_set/processed_data/SJ_labeled_selfT1NN_beta.csv")
write.csv(d_selft2, file="/research/rgs01/home/clusterHome/qtran/SJ_validation_set/processed_data/SJ_labeled_selfTkSVM_beta.csv")
write.csv(d_coBC, file="/research/rgs01/home/clusterHome/qtran/SJ_validation_set/processed_data/SJ_labeled_coBC_kSVM.csv")
write.csv(d_snnrce, file="/research/rgs01/home/clusterHome/qtran/SJ_validation_set/processed_data/SJ_labeled_SNNRCE_beta.csv")
write.csv(d_triT, file="/research/rgs01/home/clusterHome/qtran/SJ_validation_set/processed_data/SJ_labeled_triT_beta.csv")
write.csv(d_demo, file="/research/rgs01/home/clusterHome/qtran/SJ_validation_set/processed_data/SJ_labeled_demo_beta.csv")

save(m.snnrce, file= "/research/rgs01/home/clusterHome/qtran/SJ_validation_set/m.snnrce.RData")
save(m.cobc, file= "/research/rgs01/home/clusterHome/qtran/SJ_validation_set/m.cobc.RData")
save(m.trit, file= "/research/rgs01/home/clusterHome/qtran/SJ_validation_set/m.triT.RData")
save(m.demo, file= "/research/rgs01/home/clusterHome/qtran/SJ_validation_set/m.demo.RData")

