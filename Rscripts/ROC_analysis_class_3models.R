#########################
library(ROCR)
library(pROC)
main_dir = "/Volumes/qtran/Semisupervised_Learning_big_private_data/processed_data/class_results/"
fig_dir = "/Volumes/qtran/Semisupervised_Learning_big_private_data/figures/class_results/"

model = c('setred_OneNN','snnrce_OneNN', 'selfTraining_OneNN')
m = "snnrce_OneNN"
toutput = read.csv(paste0(main_dir, 'Calibrated_', m, '_Trans_scores_MCF_wide.csv'), header=TRUE)

colnames(toutput) = gsub("\\.\\.", ", ", colnames(toutput))
colnames(toutput) = gsub(".", " ", colnames(toutput), fixed=TRUE)
toutput = toutput[,-1]

######RAW MAX SCORES########
max_tscore = read.csv(paste0(main_dir, "All_MCF_Transductive_MaxScores_",m, ".csv"))
max_tscore$match = ifelse(max_tscore$pred_label == max_tscore$truth_label, 1, 0)

#####MAX CALIBRATED SCORES######
toutput$max_score = apply(toutput[, 1:91], 1, max)
toutput$max_index = apply(toutput[,1:91], 1, function(x) which.max(x))
toutput$pred_family = colnames(toutput)[toutput$max_index]
toutput$match = ifelse(toutput$truth_label == toutput$pred_family, 1, 0)

max_calScores = toutput[, c("Sample", "max_score", "truth_label", "pred_family", "seed", "match")]
colnames(max_calScores) = c("Sample", "pred_score", "truth_label", "pred_label", "seed", "match")
######create a model for binary response to get the scores for a binary SETRED-SVM classifier########
set.seed(1234)
rocobj_calb = roc(max_calScores$match, max_calScores$pred_score)
rocobj_raw = roc(max_tscore$match, max_tscore$pred_score)

thres_calb = coords(rocobj_calb, x="all",  best.method="youden", input="threshold", ret="all", transpose = FALSE)
thres = coords(rocobj_raw, x="all",  best.method="youden", input="threshold", ret="all", transpose = FALSE)

pdf(paste0(fig_dir, "/ROC_YoudenIndex_selfTraing_OneNN_Family.pdf"), width=9, height = 9)
op = par(cex=1.5, lwd=3)
plot(rocobj_calb, print.thres="best", print.thres.best.method="youden", col="blue", lwd=3, 
     auc.polygon.col="lightblue", alpha = 0.8,
     print.thres.col="blue", print.thres.adj=c(0.3,-0.6), print.thres.best.weights=c(1, 0.5), 
     print.auc=TRUE, print.auc.y = 0.77, print.auc.x = 1, 
     cex.lab=2, cex.axis=2)
plot(rocobj_raw, print.thres="best", print.thres.best.method="youden", col="red", lwd=3, 
     auc.polygon.col="lightpink", print.thres.adj=c(0.1,1.8),
     print.thres.col="red", print.auc=TRUE, add=TRUE, print.auc.y = 0.60, print.auc.x = 0.72)
legend(x=0.3,y= 0.13, legend=c("raw","calibrated"), lty = c(1,1), col=c("red","blue"), lwd=c(2,2))
dev.off()

pdf(paste0(fig_dir, "/ROC_YoudenIndex_snnrce_OneNN_Family.pdf"), width=9, height = 9)
op = par(cex=1.5, lwd=3)
plot(rocobj_calb, print.thres="best", print.thres.best.method="youden", col="blue", lwd=3, 
     auc.polygon.col="lightblue", alpha = 0.8,
     print.thres.col="blue", print.thres.adj=c(0.3,-0.6), print.thres.best.weights=c(1, 0.5), 
     print.auc=TRUE, print.auc.y = 0.84, print.auc.x = 1, 
     cex.lab=2, cex.axis=2)
plot(rocobj_raw, print.thres="best", print.thres.best.method="youden", col="red", lwd=3, 
     auc.polygon.col="lightpink", print.thres.adj=c(0.1,1.6),
     print.thres.col="red", print.auc=TRUE, add=TRUE, print.auc.y = 0.60, print.auc.x = 0.72)
legend(x=0.3,y= 0.13, legend=c("raw","calibrated"), lty = c(1,1), col=c("red","blue"), lwd=c(2,2))
dev.off()


ci.thresholds.obj <- ci.thresholds(rocobj_calb)
plot(ci.thresholds.obj, type="shape")
plot(rocobj_calb)

plot.roc(max_calScores$match, max_calScores$pred_score, ci=TRUE, of="thresholds", type="shape")

ci.se.obj <- ci.se(rocobj_calb, boot.n=1000)
ci.sp.obj <- ci.sp(rocobj_calb, boot.n=1000)



pdf(paste0(fig_dir, "Threshold_Prec_Rec_Spec_Class.pdf"), width=7, height = 7)
plot(precision ~ threshold , thres_calb, type = "l", 
     subset = is.finite(threshold), col = "blue", lwd=2,
     cex.lab=1.3, cex.axis=1.3, ylab="Precision/Recall/Specificity", ylim = c(0, 1))
lines(recall ~ threshold, thres_calb, type = "l", lty=2, 
      subset = is.finite(threshold), col = "blue", lwd=2)
lines(specificity ~ threshold, thres_calb, type = "l", lty=3, 
      subset = is.finite(threshold), col = "blue", lwd=2)
abline(v = 0.73, col="black", lwd=2, lty=2)
text(0.71, 1, "prec = 0.999", cex = 1, col = "black", adj=c(1,0))
text(0.86, 0.89, "recall = 0.894", cex = 1, col = "black", adj=c(0.67,0))
text(0.86, 0.94, "spec = 0.962", cex = 1, col = "black", adj=c(0.67,0))
legend(x=0.2, y=0.18, legend=c("precision","recall", "specificity"),
       lty = c(1,2,3), col=c("blue","blue", "blue"), lwd=c(2,2,2))
#abline(v = 0.8, col="#b7702d", lwd=2, lty=2)
#text(0.86, 0.93, "spec = 0.959", cex = 1, col = "#b7702d", adj=c(0.3,0))
#text(0.86, 0.785, "rec = 0.798", cex = 1, col = "#b7702d", adj=c(0.3,0))
#text(0.86, 1.01, "prec = 0.999", cex = 1, col = "#b7702d", adj=c(0.3,0))

dev.off()


pdf(paste0(fig_dir, "/Threshold_vs_Spec+Sens_Family.pdf"))
plot(specificity + sensitivity ~ threshold, thres_calb, type = "l",  
     subset = is.finite(threshold), col = "blue", lwd=2, ylim=c(1, 2),
     cex.lab=1.2, cex.axis=1.2)
lines(specificity + sensitivity ~ threshold, thres, type = "l",  
     subset = is.finite(threshold), col = "red", lwd=2)
abline(v = 0.73, col="blue", lwd=2, lty=2)
abline(v = 0.218, col="red", lwd=2, lty=2)
text(0.22, 1.72, "0.22", cex = 1.5, col = "red", adj=c(-0.2,0))
text(0.718, 1.83, "0.718", cex = 1.5, col = "blue", adj=c(-0.2,0))
legend(0.4,1.2, legend=c("raw","calibrated"),
       lty = c(1,1), col=c("red","blue"), lwd=c(2,2))
dev.off()

plot(rocobj_calb, print.thres="best", print.thres.best.method="youden",
     print.thres.best.weights=c(0.5, 0.5)) 


pdf(paste0(fig_dir, "/Threshold_calibrated_vs_Spec_Sens_Family.pdf"))
plot(specificity ~ threshold , thres_calb, type = "l", 
     subset = is.finite(threshold), col = "blue", lwd=2,
     cex.lab=1.3, cex.axis=1.3, ylab="Sensitivity/Specificity")
lines(sensitivity ~ threshold, thres_calb, type = "l", lty=2, 
      subset = is.finite(threshold), col = "blue", lwd=2)
abline(v = 0.71, col="black", lwd=2, lty=2)
#abline(v = 0.775, col="salmon", lwd=2, lty=2)
text(0.70, 0.9, "0.718 (0.927, 0.905)", cex = 1, col = "black", adj=c(-0.1,0))
#text(0.79, 1.01, "0.775 (0.951, 0.859)", cex = 1, col = "salmon", adj=c(0,0)) #### a different threshold
legend(x=0.2, y=0.12, legend=c("specificity","sensitivity"),
       lty = c(1,2), col=c("blue","blue"), lwd=c(2,2))
dev.off()

pdf(paste0(fig_dir, "/Prec_vs_Recall_Family.pdf"))
plot(precision ~ threshold , thres_calb, type = "l", 
     subset = is.finite(threshold), col = "blue", lwd=2,
     cex.lab=1.3, cex.axis=1.3, ylab="Precision/Recall", ylim = c(0, 1))
lines(recall ~ threshold, thres_calb, type = "l", lty=2, 
      subset = is.finite(threshold), col = "blue", lwd=2)
abline(v = 0.71, col="black", lwd=2, lty=2)
text(0.70, 1, "prec = 0.998", cex = 1, col = "black", adj=c(1,0))
text(0.70, 0.88, "recall = 0.905", cex = 1, col = "black", adj=c(1,0))
legend(x=0.2, y=0.12, legend=c("precision","recall"),
       lty = c(1,2), col=c("blue","blue"), lwd=c(2,2))
dev.off()


response = as.factor(toutput$truth_label)
thres <- rep(NA, length(levels(response)))
all_cal_scores = NULL
thres_class <- NULL

source("/Volumes/qtran/Semisupervised_Learning/draw_functions.R")

base_dir = "/Volumes/qtran/Semisupervised_Learning/figures/family_results/"
measures = c("prec", "rec", "acc", "tpr", "fpr", "fnr", "sens", "spec")
for (m in measures){
  fn = paste0(base_dir, toupper(m), "_vs_Cutoff_calibratedScores.pdf")
  thres_class = draw_cutoff_curves(max_calScores, max_calScores, colors, ylab = toupper(m), 
                     xlab= "Cutoff", title = toupper(m), metrics = m, xlim = c(0,1), ylim = c(0,1), 
                     filename=fn)
}
write.csv(thres_class, file = "/Volumes/qtran/Semisupervised_Learning/processed_data/family_results/Threshold_calibrated_scores_perMCF.csv")


draw_cutoff_curves(toutput, toutput, colors, ylab = toupper(m), 
                   xlab= "Cutoff", title = "TPR vs FPR", metrics = c("tpr", "fpr"), xlim = c(0,1), ylim = c(0,1), 
                   filename=fn)

########
library(pROC)
#######Onve vs One technique#######
mroc <- multiclass.roc(max_tscore$truth_label, max_tscore$pred_score)
auc(mroc)
rs <- mroc[['rocs']]
plot.roc(rs[[90]])
sapply(2:length(rs),function(i) lines.roc(rs[[i]],col=i))

perf <- performance(pred, metrics)
roc.x <- unlist(perf@x.values) #Recall
roc.y <- unlist(perf@y.values)#Precision
lines(roc.y ~ roc.x,  lwd = 2, col = colors$col[i])

#lines(x=c(0,1), c(0,1))
pdf(paste0(fig_dir, "Raw_vs_Calibrated_Max_Scores_Family.pdf"))
plot(density(max_tscore$pred_score), col="red", xlim=c(0,1), ylim=c(0, 8), ylab="Density", xlab="probabilities",
     main="Raw and Calibrated Probabilities", lwd=2, cex.lab=1.5, cex.axis=1.5, cex.main = 1.5)
  lines(density(max_calScores$pred_score), col="blue", lwd=2)
  legend(0,7.8, legend=c("raw","calibrated"), 
       lty = c(1,1), col=c("red","blue"), lwd=c(2,2))
dev.off()
