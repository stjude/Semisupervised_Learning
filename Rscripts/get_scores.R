library(ssc)

#load("/Volumes/qtran/Semisupervised_Learning/processed_data/seed50_allmodelsALL.RData")
load("/rgs01/home/clusterHome/qtran/Semisupervised_Learning/processed_data/seed50_allmodelsALL.RData")

#load("/rgs01/home/clusterHome/qtran/Semisupervised_Learning/label_data_x.RData")
#load("/rgs01/home/clusterHome/qtran/Semisupervised_Learning/label_data_y.RData")

m.setred = x$m.setred2
#source("/Volumes/qtran/Semisupervised_Learning/predictOneNN.R")

load("/rgs01/home/clusterHome/qtran/Semisupervised_Learning/data_sets.RData")
xtrain = x$xtrain
ytrain = x$ytrain
xitest = x$xitest
yitest = x$yitest
xttest = x$xttest
yttest = x$yttest
#dtrain <- as.matrix(proxy::dist(x = xtrain, method = "euclidean", by_rows = TRUE)) 
#ditest <- as.matrix(proxy::dist(x = xitest, y = xtrain, method = "euclidean", by_rows = TRUE))
#dttest <- as.matrix(proxy::dist(x = xttest, y = xtrain, method = "euclidean", by_rows = TRUE))

learner <- e1071::svm
learner.pars <- list(type = "C-classification", kernel="radial", 
                     probability = TRUE, scale = TRUE)
pred <- function(m, x){
  r <- predict(m, x, probability = TRUE)
  prob <- attr(r, "probabilities")
  return(prob)
}

pred_class = predict(m.setred, xitest)
tpred_class = predict(m.setred, xttest)

source("/rgs01/home/clusterHome/qtran/Semisupervised_Learning/predict_setred_prob.R")
pred_score <- predict.setred.prob (m.setred, xitest, type="prob")
tpred_score <- predict.setred.prob (m.setred, xttest, type="prob")


create_class_score_output = function(pred_label, pred_score, truth_label, truth_beta){
  output = data.frame(pred_label)
  output$Sample = rownames(truth_beta)
  output$pred_score = apply(pred_score, 1, max)
  output$truth_label = truth_label

  return(output)
}

ioutput = create_class_score_output(pred_class, pred_score, yitest, xitest)
toutput = create_class_score_output(tpred_class, tpred_score, yttest, xttest)

write.csv(ioutput, file = "/rgs01/home/clusterHome/qtran/Semisupervised_Learning/processed_data/seed50_all/Inductive_output_SETRED_SVM_seed50.csv")
write.csv(toutput, file = "/rgs01/home/clusterHome/qtran/Semisupervised_Learning/processed_data/seed50_all/Transductive_output_SETRED_SVM_seed50.csv")

#########################
toutput = read.csv("/Volumes/qtran/Semisupervised_Learning/processed_data/seed50_all/Transductive_output_SETRED_SVM_seed50.csv", header=TRUE)
ioutput = read.csv("/Volumes/qtran/Semisupervised_Learning/processed_data/seed50_all/Inductive_output_SETRED_SVM_seed50.csv", header=TRUE)

ioutput$match = factor(ifelse(ioutput$pred_label == ioutput$truth_label, "yes", "no"))

library(plyr)
imu <- ddply(ioutput, "match", summarise, grp.mean=mean(pred_score))
head(imu)

library(dplyr)
imu <- ioutput %>%
  group_by(match) %>%
  summarize(grp.mean = mean(pred_score), grp.median = median(pred_score))

library(ggplot2)
p <- ggplot(ioutput, aes(x=pred_score)) + 
  geom_histogram(binwidth = 0.05, aes(y=..density..), colour="black",fill="white", position="dodge")+
  geom_density(alpha=.2, fill="#FF6666", size=1) +
  theme_classic(base_size = 15)
p + geom_vline(aes(xintercept=mean(pred_score)),
              color="blue", linetype="dashed", size=1)

ggsave("/Volumes/qtran/Semisupervised_Learning/figures/DensityPlot_all_Ind_output_SETRED_SVM.pdf")

p1 <- ggplot(ioutput, aes(x=pred_score, color=match, fill=match)) + 
  geom_histogram(binwidth = 0.05, aes(y=..density..), position="identity", alpha=0.5) +
  geom_vline(data=imu, aes(xintercept=grp.mean, color=match),
             linetype="dashed", size = 1)+
  geom_density(alpha=.2, size=1) +
  geom_text(mapping = aes(x = grp.mean,
                          y = 4,
                          label = round(grp.mean, 2)),
            data = imu) +
  theme_minimal(base_size = 15)

#p1 + scale_color_grey()+scale_fill_grey()

p1+scale_color_brewer(palette="Set2")+
  scale_fill_brewer(palette="Set2")


estimate_mode <- function(x) {
  d <- density(x)
  d$x[which.max(d$y)]
}

estimate_mode(ioutput$pred_score)

stat_mode <- function(x, return_multiple = TRUE, na.rm = FALSE) {
  if(na.rm){
    x <- na.omit(x)
  }
  ux <- unique(x)
  freq <- tabulate(match(x, ux))
  mode_loc <- if(return_multiple) which(freq==max(freq)) else which.max(freq)
  return(ux[mode_loc])
}

stat_mode(round(ioutput$pred_score[ioutput$match=="yes"],4))
stat_mode(round(ioutput$pred_score[ioutput$match=="no"],3))

library(tidyr)
pred_long  <- gather(tpred, class, prob, 1:91, factor_key=TRUE)
library(reshape2)

#######ROC analysis######
library(pROC)

ioutput$true_labels_integer = as.integer(ioutput$truth_label)
iroc_multi = multiclass.roc(ioutput$true_labels_integer, ioutput$pred_score, levels = levels(as.factor(ioutput$true_labels_integer)))
auc(iroc_multi)
irs = iroc_multi[['rocs']]
plot.roc(irs[[84]])
sapply(2:length(irs),function(i) lines.roc(irs[[i]],col=i))

source("/research/rgs01/home/clusterHome/qtran/Semisupervised_Learning/supporting_functions.R")
iconM = createConfusionMatrix(pred_class, yitest)

irs[[84]]$thresholds[which(irs[[84]]$sensitivities + irs[[84]]$specificities == 
                            max(irs[[84]]$sensitivities + irs[[84]]$specificities))]

library(ROCR)
response = ioutput$truth_label
aucs <- rep(NA, length(levels(response))) # store AUCs
for (i in seq_along(levels(response))) {
  cur.class <- levels(response)[i]
  binary.labels <- as.factor(ioutput$truth_label == cur.class)
  # binarize the classifier you are using (NB is arbitrary)
  #model <- NaiveBayes(binary.labels ~ ., data = iris.train[, -5])
  #pred <- predict(model, iris.test[,-5], type='raw')
  score <- ioutput$pred_score[test.labels] # posterior for  positive class
  test.labels <- ioutput$truth_label == cur.class
  pred <- prediction(score, test.labels)
  perf <- performance(pred, "prec", "rec")
  roc.x <- unlist(perf@x.values)
  roc.y <- unlist(perf@y.values)
  lines(roc.y ~ roc.x, col = colors[i], lwd = 2)
  # store AUC
  auc <- performance(pred, "auc")
  auc <- unlist(slot(auc, "y.values"))
  aucs[i] <- auc
}
lines(x=c(0,1), c(0,1))
legend("bottomright", levels(response), lty=1, 
       bty="n", col = colors)