library(ssc)

load("/Volumes/qtran/Semisupervised_Learning/m.setred.svm.seed50.RData")
load("/Volumes/qtran/Semisupervised_Learning/data_sets.RData")

load("/rgs01/home/clusterHome/qtran/Semisupervised_Learning/data_sets.RData")
xtrain = x$xtrain
ytrain = x$ytrain
xitest = x$xitest
yitest = x$yitest
xttest = x$xttest
yttest = x$yttest

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

#source("/rgs01/home/clusterHome/qtran/Semisupervised_Learning/predict_setred_prob.R")
source("/Volumes/qtran/Semisupervised_Learning/predict_setred_prob.R")

pred_score <- predict.setred.prob (m.setred, xitest, type="prob")
tpred_score <- predict.setred.prob (m.setred, xttest, type="prob")

library(rfUtilities)
calibrated.y.hat <- probability.calibration(yitest, pred_score, regularization = TRUE) 

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
write.csv(pred_score,file = "/Volumes/qtran/Semisupervised_Learning/processed_data/seed50_all/Inductive_scores_SETRED_SVM_seed50.csv")
write.csv(tpred_score,file = "/Volumes/qtran/Semisupervised_Learning/processed_data/seed50_all/Transductive_scores_SETRED_SVM_seed50.csv")

#########################
toutput = read.csv("/Volumes/qtran/Semisupervised_Learning/processed_data/seed50_all/Transductive_output_SETRED_SVM_seed50.csv", header=TRUE)
ioutput = read.csv("/Volumes/qtran/Semisupervised_Learning/processed_data/seed50_all/Inductive_output_SETRED_SVM_seed50.csv", header=TRUE)

ioutput$match = factor(ifelse(ioutput$pred_label == ioutput$truth_label, "yes", "no"))
toutput$match = factor(ifelse(toutput$pred_label == toutput$truth_label, "yes", "no"))

library(dplyr)
imu <- toutput %>%
  group_by(match) %>%
  summarize(grp.mean = mean(pred_score), grp.median = median(pred_score))

library(ggplot2)
p <- ggplot(toutput, aes(x=pred_score)) + 
  geom_histogram(binwidth = 0.05, aes(y=..density..), colour="black",fill="white", position="dodge")+
  geom_density(alpha=.2, fill="#FF6666", size=1) +
  theme_classic(base_size = 15)
p + geom_vline(aes(xintercept=mean(pred_score)),
              color="blue", linetype="dashed", size=1)

ggsave("/Volumes/qtran/Semisupervised_Learning/figures/DensityPlot_all_Ind_output_SETRED_SVM.pdf")

p1 <- ggplot(toutput, aes(x=pred_score, color=match, fill=match)) + 
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
ggsave("/Volumes/qtran/Semisupervised_Learning/figures/DensityPlot_Ind_byMatched_SETRED_SVM.pdf")


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

#######ROC analysis######
pred_score_int = pred_score
colnames(pred_score_int) = seq(1:91)
###########ONE-VS-ONE Performance
library(pROC)

ioutput$true_labels_integer = as.integer(as.factor(ioutput$truth_label))
#give the ROC of all 4095 pairwise comparisons
iroc_multi = multiclass.roc(ioutput$truth_label, pred_score, direction = ">")
auc(iroc_multi)
irs = iroc_multi[['rocs']]

source("/research/rgs01/home/clusterHome/qtran/Semisupervised_Learning/supporting_functions.R")
source("/Volumes/qtran/Semisupervised_Learning/supporting_functions.R")
iconM = createConfusionMatrix(pred_class, yitest, mode="everything", positive = NULL)
pair_thres = NULL
k =0
for (i in irs) {
  if(k ==0){
    pair_thres = coords(i[[1]], x="best",  best.method="closest.topleft", input="specificity",
                        ret=c("threshold", "specificity", "sensitivity", "accuracy",
                              "precision", "recall", "youden", "closest.topleft"))
    k = k+1
    plot(i[[1]], print.thres="best", print.thres.best.method="youden") 
    rownames(pair_thres) = names(irs)[k]
  }else{
    k = k+1
    thres = coords(i[[1]], x="best",  best.method="closest.topleft", input = "specificity",
                   ret=c("threshold", "specificity", "sensitivity", "accuracy",
                         "precision", "recall", "youden", "closest.topleft"))
    if(nrow(thres) > 1){
      num_r = nrow(thres)
      rownames(thres) = paste0(names(irs)[k], "_", seq(num_r))
    }else{
      rownames(thres) = names(irs)[k]
    }
    pair_thres = rbind(pair_thres, thres)
  }
}

write.csv(pair_thres, file = "/Volumes/qtran/Semisupervised_Learning/processed_data/Threshold_pairwise_BestSpec_transductive.csv")
################ONE-VS-ALL Performance#########
library(ROCR)
colors = read.csv("/Volumes/qtran/Capper_validation_set/processed_data/color_key_allRef.csv", header=TRUE)
source("/Volumes/qtran/Capper_validation_set/draw_functions.R")
response = as.factor(ioutput$truth_label)
aucs <- rep(NA, length(levels(response))) # store AUCs
thres <- rep(NA, length(levels(response)))
plot(x=NA, y=NA, xlim=c(0, 1), ylim=c(0,1),
     ylab="Precision",
     xlab="Recall", main = "", 
     bty='n')

class_thres <- NULL
for (i in seq_along(levels(response))) {
  cur.class <- levels(response)[i]
  binary.labels <- as.factor(toutput$truth_label == cur.class)
  score <- tpred_score[, cur.class] # scores for all samples at the particular class
  test.labels <- toutput$truth_label == cur.class
  rocobj = roc(test.labels, score)
  
  thres = coords(rocobj, x="best",  best.method="closest.topleft", input="specificity",
         ret=c("threshold", "specificity", "sensitivity", "accuracy",
               "precision", "recall", "youden", "closest.topleft"))
  if(nrow(thres) > 1){
    num_r = nrow(thres)
    rownames(thres) = paste0(cur.class, "_", seq(num_r))
  }else{
    rownames(thres) = cur.class
  }
  class_thres = rbind(class_thres, thres)
  
  pred <- prediction(score, test.labels)
  perf <- performance(pred, "rec", "prec")
  roc.x <- unlist(perf@x.values) #Recall
  roc.y <- unlist(perf@y.values)#Precision
  lines(roc.y ~ roc.x,  lwd = 2, col = colors$col[i])
}

iclass_thres = draw_curves(ioutput, pred_score, colors, ylab = "Precision", 
            xlab= "Cutoff", title = "Precision", metrics = "prec", xlim = c(0,1), ylim = c(0,1))
tclass_thres = draw_curves(toutput, tpred_score, colors, ylab = "Precision", 
            xlab= "Cutoff", title = "Precision", metrics = "prec", xlim = c(0,1), ylim = c(0,1))

write.csv(class_thres, file = "/Volumes/qtran/Semisupervised_Learning/processed_data/Threshold_byClass_BestSpec_SETREDSVM_Transductive.csv")
  
  pred <- prediction(score, test.labels)
  perf <- performance(pred, "cal")
  roc.x <- unlist(perf@x.values)
  roc.y <- unlist(perf@y.values)
  lines(roc.y ~ roc.x,  lwd = 2, col = colors$col[i])
  # store AUC
  tfr <- performance(pred, "fpr")
  plot(tfr,xlim=c(0,0.015),ylim=c(0,1), lwd=2,
       text.cex=0.8)
  

measures <- c('tpr', 'fpr', 'prec', 'rec')
par(mfrow=c(2,2))
par(mar=c(2,2,2,2))
for (i in 1:(length(measures)-1)) {
  print(i)
  for (j in (i+1):length(measures)) {
    print(paste("HERE", measures[j]))
    perf <- performance(pred, measures[i], measures[j])
    plot(perf, avg="threshold", colorize=TRUE,
         xlim=c(0,1),ylim=c(0,1), lwd=1.5,
         text.col= as.list(terrain.colors(10)), 
         points.col= as.list(terrain.colors(10))
    )
  }
}

lines(x=c(0,1), c(0,1))
legend("bottom", levels(response), lty=1, bty="n", col = colors$col)
print(paste0("Mean AUC under the precision-recall curve is: ", round(mean(aucs), 4)))



