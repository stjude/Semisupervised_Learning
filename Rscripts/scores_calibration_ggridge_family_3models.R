
colors.family = read.csv("/Volumes/qtran/Capper_reference_set/colors.family.key.csv")
colors.family = read.csv("/research/rgs01/home/clusterHome/qtran/Capper_reference_set/colors.family.key.csv")

main_dir = "/research/rgs01/home/clusterHome/qtran/Semisupervised_Learning_big_private_data/processed_data/family_results/"
fig_dir = "/research/rgs01/home/clusterHome/qtran/Semisupervised_Learning_big_private_data/figures/family_results/"
base_dir = "/research/rgs01/home/clusterHome/qtran/Semisupervised_Learning_big_private_data/"
#base_dir = "/Volumes/qtran/Semisupervised_Learning_big_private_data/figures/family_results/"
#main_dir = "/Volumes/qtran/Semisupervised_Learning_big_private_data/processed_data/family_results/"
library(glmnet)
library(doMC)
library(tidyr)
registerDoMC(8)

model = c('setred_OneNN','snnrce_OneNN', 'selfTraining_OneNN')
for(m in model){
  tscores = read.csv(paste0(main_dir, "All_MCF_", m, "_Transductive_scores_wide.csv"), header=TRUE)
  #iscores = read.csv("/Volumes/qtran/Semisupervised_Learning/processed_data/family_results/All_MCF_SETRED_SVM_Inductive_scores_wide.csv", header=TRUE)
  
  traw = read.csv(paste0(main_dir, "All_MCF_", m, "_Transductive_scores_long.csv"), header=TRUE)
  #iraw = read.csv("/Volumes/qtran/Semisupervised_Learning/processed_data/family_results/All_MCF_SETRED_SVM_Inductive_scores_long.csv", header=TRUE)
  
  ########
  ####Use the predictive scores from the transductive data sets for glmnet calibration
  ####Then apply the model with the best lambda to get the calibrated transductive scores
  set.seed(1234)
  fit = glmnet(as.matrix(tscores[ ,2:76]), as.factor(tscores$truth_label), family = "multinomial", 
               type.multinomial = "ungrouped", alpha=0, trace.it = 1)  #alpha =0 ridge penalty
  #fit
  #plot(fit, xvar = "lambda", label = TRUE, type.coef = "2norm")
  set.seed(1234)
  cvfit <- cv.glmnet(as.matrix(tscores[ ,2:76]), as.factor(tscores$truth_label),
                     family = "multinomial", type.multinomial = "ungrouped", alpha=0, trace.it = 1, parallel = TRUE)
  #plot(cvfit)
  best.lambda <- cvfit$lambda.min
  write.table(best.lambda, file = paste0(base_dir, "lambda_for_calibration_model_", m, "_family.txt"))
  save(fit, file =paste0(base_dir, "calibration_model_", m, "_family.RData"))
  
  pfit = predict(fit, as.matrix(tscores[,2:76]), s = best.lambda, type = "response")
  #plot(x=pfit, tscores$truth_label)
  
  pfit = as.data.frame(pfit)
  pfit$Sample = tscores$Sample
  pfit$truth_label = tscores$truth_label
  pfit$seed = tscores$seed
  colnames(pfit) = gsub(".1", "", colnames(pfit), fixed=TRUE)
  write.csv(pfit, file = paste0(main_dir, "Calibrated_", m, "_Trans_scores_MCF_wide.csv"))
  
  calibrated_scores <- gather(pfit, pred_family, pred_score, 1:75, factor_key=FALSE)  
  
  #calibrated_scores$pred_family = gsub("\\.1", "", calibrated_scores$pred_family)
  calibrated_scores$match = ifelse(calibrated_scores$truth_label == calibrated_scores$pred_family, "yes", "no")
  write.csv(calibrated_scores, file = paste0(main_dir, "Calibrated_", m, "_Trans_scores_MCF_long.csv"), row.names = TRUE)
}
####draw density of all classes#####
library(ggplot2)

calibrated_scores$col = colors.family$color[match(calibrated_scores$truth_label, colors.family$group)]
calibrated_scores$truth_label = as.factor(calibrated_scores$truth_label)

library(ggridges)
nomatched = calibrated_scores[calibrated_scores$match == "no",]
p <-  ggplot(nomatched, aes(pred_score, truth_label, fill = truth_label))+
  labs(title="Calibrated scores for incorrectly classified cases", x="", y="")+
  scale_x_continuous(limits = c(-0.0005, 0.4), breaks = seq(0, 1, 0.1)) +
  scale_fill_manual(values = colors.family$color) +
  geom_density_ridges(alpha= 0.8,  scale = 4, rel_min_height = 0.01) +
  theme_ridges(font_size=15)+
  theme(axis.text.y = element_text(colour = colors.family$color),
        axis.text.x = element_text(size=15))+
  theme(legend.position="none")
ggsave(p, file = paste0(fig_dir, "Density_calibrated_", m,"_scores_Unclassifiable_cases.pdf"))

matched = calibrated_scores[calibrated_scores$match == "yes",]
p1 = ggplot(matched, aes(pred_score, as.factor(truth_label), fill = truth_label, group=truth_label))+
  labs(title="Calibrated scores for correctly classified cases", x="", y="")+
  scale_x_continuous(limits = c(-0.0050, 1), breaks = seq(0, 1, 0.1)) +
  scale_fill_manual(values = colors.family$color) +
  geom_density_ridges(alpha= 0.8, rel_min_height = 0.01, scale = 4, na.rm=TRUE) +
  theme_ridges(font_size=12)+
  theme(axis.text.y = element_text(colour = colors.family$color),
        axis.text.x = element_text(size=12))+
  theme(legend.position="none")
ggsave(p1, file = paste0(fig_dir, "Density_calibrated_scores_Classifiable_cases.pdf"))

#####################RAW_SCORES####################
traw$truth_label = as.factor(traw$truth_label)
nomatched_raw = traw[traw$match == 0,]
p2 = ggplot(nomatched_raw, aes(pred_score, truth_label, fill = truth_label))+
  labs(title="Raw transductive scores for unclassifiable cases", x="", y="")+
  scale_x_continuous(limits = c(-0.0005, 0.4), breaks = seq(0, 1, 0.1)) +
  scale_fill_manual(values = colors.family$color) +
  geom_density_ridges(alpha= 0.8,  scale = 4, rel_min_height = 0.001, na.rm=TRUE) +
  theme_ridges(font_size=12)+
  theme(axis.text.y = element_text(colour = colors.family$color),
        axis.text.x = element_text(size=12),
        plot.title = element_text(size=12))+
  theme(legend.position="none")
ggsave(p2, file = paste0(fig_dir, "Density_raw_scores_Unclassifiable_cases.pdf"))


traw_matched = traw[traw$match == "1",]
p3 = ggplot(traw_matched, aes(pred_score, truth_label, fill = truth_label, group=truth_label))+
  labs(title="Raw transductive scores for correctly classified cases", x="", y="")+
  scale_x_continuous(limits = c(-0.0050, 0.1), breaks = seq(0, 1, 0.1)) +
  scale_fill_manual(values = colors.family$color) +
  geom_density_ridges(alpha= 0.8, rel_min_height = 0.001, scale = 4, na.rm=TRUE) +
  theme_ridges(font_size=12)+
  theme(axis.text.y = element_text(colour = colors.family$color),
        axis.text.x = element_text(size=12),
        plot.title = element_text(size=12))+
  theme(legend.position="none")
ggsave(p3, file = paste0(fig_dir, "Density_raw_scores_Classifiable_cases.pdf"))
