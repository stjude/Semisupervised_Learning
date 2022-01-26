tscores = read.csv("/Volumes/qtran/Semisupervised_Learning/processed_data/class_results/All_SETRED_SVM_Transductive_scores_wide.csv", header=TRUE)
iscores = read.csv("/Volumes/qtran/Semisupervised_Learning/processed_data/family_results/All_SETRED_SVM_Inductive_scores_wide.csv", header=TRUE)
base_dir = "/Volumes/qtran/Semisupervised_Learning/figures/class_results/"
main_dir = "/Volumes/qtran/Semisupervised_Learning/processed_data/"
traw = read.csv("/Volumes/qtran/Semisupervised_Learning/processed_data/class_results/All_SETRED_SVM_Transductive_scores_long.csv", header=TRUE)
iraw = read.csv("/Volumes/qtran/Semisupervised_Learning/processed_data/class_results/All_SETRED_SVM_Inductive_scores_long.csv", header=TRUE)

colors = read.csv("/Volumes/qtran/Capper_reference_set/color.key.allRef.csv")
which(duplicated(tscores$Sample))

####get all train data####
#We want to get the unlabel samples that are not in the labeled samples across 7 resampling
#####
get_allTrain_unlabeled_samples = function(folder, type=c("class_results", "family_results")){
  allTrain = NULL
  for (s in c("seed1", "seed2", "seed20", "seed40", "seed80", "seed160", "seed320")){
    print(paste0("Start with seed = ", s))
    filename = paste0(folder, type, "/", s, "_alldata_sets.RData")
    load(filename)
    #d = rownames(x$xtrain)
    t = rownames(x$xttest) ###unlabel samples
    #l = setdiff(d, t)
    if (s == "seed1"){
      allTrain = t
    }else{
      allTrain = intersect(allTrain, t)
    }
  }
  return(allTrain)
}

########
####Use the predictive scores from the transductive data sets for glmnet calibration
####Then apply the model with the best lambda to get the calibrated transductive scores
library(glmnet)
library(doMC)
registerDoMC(4)
fit = glmnet(as.matrix(tscores[tscores$seed == "seed1_all" ,2:92]), as.factor(tscores$truth_label[tscores$seed=="seed1_all"]), family = "multinomial", 
             type.multinomial = "ungrouped", alpha=0, trace.it = 1) 

fit = glmnet(as.matrix(tscores[,2:92]), as.factor(tscores$truth_label), family = "multinomial", 
             type.multinomial = "ungrouped", alpha=0, trace.it = 1)  #alpha =0 ridge penalty
fit
plot(fit, xvar = "lambda", label = TRUE, type.coef = "2norm")
cvfit <- cv.glmnet(as.matrix(tscores[tscores$seed == "seed2_all",2:92]), as.factor(tscores$truth_label[tscores$seed=="seed2_all"]),
                   family = "multinomial", type.multinomial = "ungrouped", alpha=0, trace.it = 1, parallel = TRUE)
cvfit <- cv.glmnet(as.matrix(tscores[,2:92]), as.factor(tscores$truth_label),
                   family = "multinomial", type.multinomial = "ungrouped", alpha=0, trace.it = 1, parallel = TRUE)
plot(cvfit)
best.lambda <- cvfit$lambda.min

cv.pfit = predict(cvfit, as.matrix(tscores[,2:92]), s = best.lambda, type = "response")
pfit = predict(fit, as.matrix(tscores[,2:92]), s = best.lambda, type = "response")

#plot(x=pfit, tscores$truth_label)

pfit = as.data.frame(pfit)
pfit$Sample = tscores$Sample
pfit$truth_label = tscores$truth_label
pfit$seed = tscores$seed
colnames(pfit) = gsub(".1", "", colnames(pfit), fixed=TRUE)
write.csv(pfit, file = "/Volumes/qtran/Semisupervised_Learning/processed_data/class_results/Calibrated_Trans_scores_wide.csv")

library(tidyr)
calibrated_scores <- gather(pfit, pred_label, pred_score, 1:91, factor_key=FALSE)  

calibrated_scores$pred_label = gsub("\\.1", "", calibrated_scores$pred_label)
calibrated_scores$match = ifelse(calibrated_scores$truth_label == calibrated_scores$pred_label, "yes", "no")
write.csv(calibrated_scores, file = "/Volumes/qtran/Semisupervised_Learning/processed_data/class_results/Calibrated_Trans_scores_long.csv")
calibrated_scores = read.csv(file = "/Volumes/qtran/Semisupervised_Learning/processed_data/class_results/Calibrated_Trans_scores_long.csv", header=TRUE)
####draw density of all classes#####
library(ggplot2)

calibrated_scores$col = colors$color[match(calibrated_scores$truth_label, colors$title.class)]
calibrated_scores$truth_label = as.factor(calibrated_scores$truth_label)

library(ggridges)
no_matched = calibrated_scores[calibrated_scores$match == "no",]
p <- ggplot(no_matched, aes(pred_score, truth_label, fill = truth_label))+
  labs(title="Calibrated scores for incorrectly classified cases", x="", y="")+
  scale_x_continuous(limits = c(-0.0005, 0.4), breaks = seq(0, 1, 0.1)) +
  scale_fill_manual(values = colors$color) +
  geom_density_ridges(alpha= 0.8,  scale = 4, rel_min_height = 0.01, na.rm=TRUE) +
  theme_ridges(font_size=12)+
  theme(axis.text.y = element_text(colour = colors$color),
        axis.text.x = element_text(size=12))+
  theme(legend.position="none")
ggsave(p, file = paste0(base_dir, "Density_calibrated_scores_Unclassifiable_cases.pdf"), width=4, height=6.5)

matched = calibrated_scores[calibrated_scores$match == "yes",]
p1 = ggplot(matched, aes(pred_score, as.factor(truth_label), fill = truth_label, group=truth_label))+
  labs(title="Calibrated scores for correctly classified cases", x="", y="")+
  scale_x_continuous(limits = c(-0.0050, 1), breaks = seq(0, 1, 0.1)) +
  scale_fill_manual(values = colors$color) +
  geom_density_ridges(alpha= 0.8, rel_min_height = 0.01, scale = 4, na.rm=TRUE) +
  theme_ridges(font_size=12)+
  theme(axis.text.y = element_text(colour = colors$color),
        axis.text.x = element_text(size=12))+
  theme(legend.position="none")
ggsave(p1, file = paste0(base_dir, "Density_calibrated_scores_Classifiable_cases.pdf"), width=4, height=6.5)

#####################RAW_SCORES####################
traw$truth_label = as.factor(traw$truth_label)

p2 =  ggplot(traw[traw$match=="0", ], aes(pred_score, truth_label, fill = truth_label))+
  labs(title="Raw transductive scores for unclassifiable cases", x="", y="")+
  scale_x_continuous(limits = c(-0.0005, 0.04), breaks = seq(0, 1, 0.01)) +
  scale_fill_manual(values = colors$color) +
  geom_density_ridges(alpha= 0.8,  scale = 4, rel_min_height = 0.001, na.rm=TRUE) +
  theme_ridges(font_size=12)+
  theme(axis.text.y = element_text(colour = colors$color),
        axis.text.x = element_text(size=12),
        plot.title = element_text(size=12))+
  theme(legend.position="none")
ggsave(p2, file = paste0(base_dir, "Density_raw_scores_Unclassifiable_cases.pdf"))


traw_matched = traw[traw$match == "1",]
p3 = ggplot(traw_matched, aes(pred_score, truth_label, fill = truth_label, group=truth_label))+
  labs(title="Raw transductive scores for correctly classified cases", x="", y="")+
  scale_x_continuous(limits = c(-0.0050, 1), breaks = seq(0, 1, 0.1)) +
  scale_fill_manual(values = colors$color) +
  geom_density_ridges(alpha= 0.8, rel_min_height = 0.001, scale = 4, na.rm=TRUE) +
  theme_ridges(font_size=12)+
  theme(axis.text.y = element_text(colour = colors$color),
        axis.text.x = element_text(size=12),
        plot.title = element_text(size=12))+
  theme(legend.position="none")
ggsave(p3, file = paste0(base_dir, "Density_raw_scores_Classifiable_cases.pdf"))
