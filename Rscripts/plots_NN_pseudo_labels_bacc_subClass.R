library(ggplot2)
library(glmnet)

pseudo_result = read.csv('./python/output/all_seed_pseudo_label_subclass_summary.csv', header=TRUE)
pseudo_result = pseudo_result[,-1]
colnames(pseudo_result) = gsub("\\.\\.", ', ', colnames(pseudo_result))
colnames(pseudo_result) = gsub("\\.", ' ', colnames(pseudo_result))

#####
select_dataset = c("35", "70_GSE", "LC", "HC")

ref_pseudo_count = read.csv('./python/output/ref_pseudo_labels_count_subClass.csv', header=TRUE)
ref_pseudo_count$Label.type = factor(ref_pseudo_count$Label.type, 
                                     levels = c("HC SS label", "HC gse SS label", "SS label","gse SS label", 
                                                "reference label", "test reference label"))
ref_pseudo_count$Label_type = factor(ref_pseudo_count$Label_type, 
                                     levels = c("HC SS labels",  "LC SS labels", "Reference labels"))
ref_pseudo_count$Dataset = factor(ref_pseudo_count$Dataset, levels=c("35", "35_GSE", "70_GSE","70HC_GSE", "70", "70_gseHC",  
                                                                     "35_gseHC", "70HC","HC"))

prop_fig = ggplot(ref_pseudo_count[1:22,], aes(fill=Label_type, x=Dataset, y = Proportion3)) +
  geom_bar(position="stack", stat="identity" ) +
  scale_fill_brewer(palette = "Set3")+
  ylab("Proportion")+
  geom_text(aes(y = label_y, label = round(Proportion3, digits=2)), vjust = 1.5, colour = "black", size=8)+
  #scale_fill_manual(values = c("#8DD3C7", "#FFFFB3", "#80B1D3"))+
  scale_y_continuous(breaks = seq(0, 1, by = .25)) +
  scale_x_discrete(position="top")+
  theme_bw(base_size = 22, base_rect_size = 2, base_line_size = 1.8)+
  theme(axis.text.x = element_blank(),
        axis.title.x = element_blank(),
        axis.title.y = element_text(size = 25, face="bold"),
        axis.text.y = element_text(color = "black", size = 25, face="bold"),
        panel.grid.minor = element_blank(),
        panel.grid.major = element_blank(),
        legend.text=element_text(size=25),
        legend.position = "none")
prop_fig
ggsave(prop_fig, file="./figures/proportion3_pseudo_labels_subClass.pdf")

count_fig = ggplot(ref_pseudo_count[1:22,], aes(fill=Label_type, x=Dataset, y = Count)) +
  geom_bar(position="stack", stat="identity" ) +
  scale_fill_brewer(palette = "Set3")+
  ylab("Count")+
  #scale_fill_manual(values = c("#8DD3C7", "#FFFFB3", "#80B1D3"))+
  geom_text(aes(y = label_y2, label = Count), vjust = 1.5, colour = "black", size=8)+
  scale_y_continuous(breaks = seq(0, 3000, by = 500), position="right") +
  scale_x_discrete(position="top")+
  theme_bw(base_size = 22, base_rect_size = 2, base_line_size = 1.8)+
  theme(axis.text.x = element_blank(),
        axis.text.y = element_text(color = "black", size=25, face = "bold"),
        axis.title.x = element_blank(),
        axis.title.y = element_text(size = 25, face="bold"),
        panel.grid.minor = element_blank(),
        panel.grid.major = element_blank(),
        legend.title = element_blank(),
        legend.text=element_text(size=27, face="bold"),
        legend.position = c(0.80, 0.88))
count_fig
ggsave(count_fig, file="./figures/count2_pseudo_labels_subClass.pdf")

#####

pseudo_long <- gather(pseudo_result, meth_class, count, 3:93, factor_key=TRUE)
pseudo_long$Dataset = trimws(pseudo_long$Dataset)
pseudo_l20 = pseudo_long[pseudo_long$count<20,]
pseudo_g20l40 = pseudo_long[20<pseudo_long$count<40,]
pseudo_g40 = pseudo_long[pseudo_long$count>40,]

pseudo_mean <- pseudo_long %>%
  group_by(meth_class, Dataset) %>%
  summarize(Mean = mean(count, na.rm=TRUE))

high_freq_class = c("A IDH", "DMG, K27", "EPN, MPE", "EPN, PF A", "EPN, PF B", "EPN, RELA",
                    "GBM, MES", "GBM, RTK I", "GBM, RTK II", "LGG, PA PF", "MB, G3", "MB, G4",
                    "MB, SHH CHL AD", "MNG", "O IDH")
inter_freq_class = c("A IDH, HG", "ANA PA", "ATRT, MYC", "ATRT, SHH", "ATRT, TYR", "CNS NB, FOXR2",
                     "EPN, SPINE", "ETMR", "GBM, G34", "HMB",
                      "LGG, DNT", "LGG, PA GG ST", "LGG, PA MID", 
                     "MB, SHH INF", "MB, WNT", 
                     "PITUI", "PLEX, PED B", "PXA", "SCHW", "SUBEPN, PF")

HFfig = ggplot(pseudo_mean[pseudo_mean$meth_class %in% high_freq_class,], aes(x=as.factor(Dataset), y=Mean, fill=Dataset)) + 
  geom_bar(stat='identity') + scale_fill_brewer(palette = "Set3") +  
  facet_wrap(~meth_class)+
  theme_bw(base_size = 20)+
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1),
        axis.title.x = element_blank(),
        panel.grid.minor = element_blank(), panel.grid.major = element_blank())
print(HFfig + ggtitle("Average count of high frequency methylation classes"))
ggsave(HFfig, file = "./figures/class_results/pseudo_labels_count_HFclasses.pdf")

IFfig = ggplot(pseudo_mean[pseudo_mean$meth_class %in% inter_freq_class,], aes(x=as.factor(Dataset), y=Mean, fill=Dataset)) + 
  geom_bar(stat='identity') + scale_fill_brewer(palette = "Set3") +  
  facet_wrap(~meth_class)+
  theme_bw(base_size = 20)+
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1),
        axis.title.x = element_blank(),
        panel.grid.minor = element_blank(),
        panel.grid.major = element_blank())
print(IFfig + ggtitle("Average count of intermediate frequency methylation classes"))
ggsave(IFfig, file = "./figures/class_results/pseudo_labels_count_IFclasses.pdf")


LFfig = ggplot(pseudo_mean[!(pseudo_mean$meth_class %in% high_freq_class |
                               pseudo_mean$meth_class %in% inter_freq_class),], 
               aes(x=as.factor(Dataset), y=Mean, fill=Dataset)) + 
  geom_bar(stat='identity') + scale_fill_brewer(palette = "Set3") +  
  facet_wrap(~meth_class)+
  theme_bw(base_size = 20)+
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1),
        axis.title.x = element_blank(),
        panel.grid.minor = element_blank(), panel.grid.major = element_blank())
print(LFfig + ggtitle("Average count of low frequency methylation classes"))
ggsave(LFfig, file = "./figures/class_results/pseudo_labels_count_LFclasses.pdf")


########
result = read.csv('./python/output/all_seeds_RF_balanced_subclass_summary.csv', header=TRUE)

result = read.csv('./python/output/all_seeds_RF+NN_balanced_subclass_summary.csv', header=TRUE)
########ANOVA#########

CV = result[result$Validation == "cross_val", ]
CV_NN = CV[CV$Model == "NN",]
CV_NN$Metric[which(CV_NN$Metric == "accuracy")] = "balanced_acc"
CV_RF = CV[CV$Model == "RF",]
CV = rbind(CV_NN, CV_RF)
HO = result[result$Validation == "vs_testset",]
bacc_CV = CV[CV$Metric == "balanced_acc",]
bacc_HO = HO[HO$Metric == "balanced_acc",]

get_pw_test = function(data, model = c("RF", "NN")){
  sub_data = data[data$Model == model, ]
  aov = aov(Value ~ as.factor(Dataset), data = sub_data)
  #pairwise.t.test(bacc_CV$Value, bacc$Dataset, p.adj = "fdr")
  pairwise_family = as.data.frame(TukeyHSD(aov)$`as.factor(Dataset)`)
  return(pairwise_family)
}

for (m in c("RF", "NN")){
  print(m)
  cv_temp = bacc_CV[bacc_CV$Dataset %in% select_dataset,]
  ho_temp = bacc_HO[bacc_HO$Dataset %in% select_dataset,]
  cv_pw = get_pw_test(cv_temp, m)
  ho_pw = get_pw_test(ho_temp, m)
  write.csv(cv_pw, file =paste0("./processed_data/class_results/", m, "_bacc_CV_TukeyHSD_subclass_4groups.csv"))
  write.csv(ho_pw, file =paste0("./processed_data/class_results/", m, "_bacc_HO_TukeyHSD_subclass_4groups.csv"))
}

metric_names <- c(
  `accuracy` = "ACCURACY",
  `balanced_acc` = "BALANCED ACCURACY",
  `recall_weighted` = "WEIGHTED RECALL",
  `precision` = "PRECISION",
  `recall` = "RECALL",
  `recall_weighted` = "RECALL",
  `precision_weighted` = "PRECISION",
  `cross_val` = "Cross Validation",
  `vs_testset` = "Hold-out Test",
  `RF` = "Random Forest",
  `NN` = "Neural Net"
  
)

#res_acc = result[result$Metric == "balanced_acc",]
res_acc = rbind(bacc_CV, bacc_HO)
res_acc$Model = factor(res_acc$Model, levels=c("RF", "NN"))
select_dataset = c("35", "70_GSE", "HC", "LC")
res_acc = res_acc[res_acc$Dataset %in% select_dataset,]
#res_acc$Metric = ifelse(res_acc$Metric == "accuracy", "balanced_acc", "balanced_acc")
#sub_data = res_acc[!(res_acc$Dataset %in% c("35_gseHC", "35_GSE")),]
#sub_data$Dataset = as.character(sub_data$Dataset)
#sub_data$Dataset = factor(sub_data$Dataset, levels=c("35", "70", "70_GSE", "70HC", "70HC_GSE", "70_gseHC", "HC"))
res_acc$Dataset = factor(res_acc$Dataset, levels=c("35", "35_GSE", "70_GSE","70HC_GSE", "70", "70_gseHC",  
                                                   "35_gseHC", "70HC", "HC"))
res_acc$Dataset = factor(res_acc$Dataset, levels = c("35", "70_GSE", "LC", "HC"))
fig1 = ggplot(res_acc, aes(x=Dataset, y=Value, fill=Dataset)) + 
  geom_boxplot(outlier.shape = NA) + #geom_jitter(width=0.3)+
  ylab("Balanced Accuracy")+
  facet_grid(Validation~Model, labeller = as_labeller(metric_names))+
  theme_bw(base_size = 22)+
  scale_x_discrete(labels=c(`35` = "35% reference",
                   `70_GSE` = "+ all SS labels",
                   `LC` = "+ LC SS labels",
                   `HC` = "+ HC SS labels")) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1, color = "black", face="bold"),
        axis.text.y = element_text(color="black", face="bold"),
        axis.title.x = element_blank(),
        strip.text.x = element_text(size = 23),
        panel.grid.minor = element_blank(),
        legend.position = "none")
fig1
ggsave(fig1, file = './figures/class_results/RF_NN_overall_balanced_ACC-subClass_v3.pdf')

RF_test_CV = aov(Value ~ Dataset, data = res_acc[res_acc$Model == "RF" & res_acc$Validation == "cross_val",])
RF_pw_CV = TukeyHSD(RF_test_CV)
write.csv(RF_pw_CV$Dataset, file = "./processed_data/class_results/RF_TukeyHSD_CV_result.csv")
RF_test_HO = aov(Value ~ Dataset, data = res_acc[res_acc$Model == "RF" & res_acc$Validation != "cross_val",])
write.csv(TukeyHSD(RF_test_HO)$Dataset, file = "./processed_data/class_results/RF_TukeyHSD_HO_result.csv")
#######
res_pr = result[result$Metric %in% c("precision", "recall", "recall_weighted", "precision_weighted"),]
res_pr$Metric = as.character(res_pr$Metric)
res_pr$Metric= ifelse(res_pr$Metric == "recall_weighted", "recall",
                      ifelse(res_pr$Metric =="precision_weighted", "precision", res_pr$Metric))
fig2 = ggplot(res_pr, aes(x=Dataset, y=Value, fill=Dataset)) + 
  geom_boxplot(outlier.shape = NA) + #geom_jitter(width=0.3)+
  facet_grid(Metric~Validation, labeller = as_labeller(metric_names))+
  theme_bw(base_size = 16)+
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1),
        axis.title.x = element_blank(),
        panel.grid.minor = element_blank())
fig2
ggsave(fig2, file = './figures/class_results/RF_overall_balanced_PreRecall-class.pdf')

####


###################
result_perClass = read.csv('./python/output_acc_perClass/all_seeds_NN_bacc_perClass.csv', header=TRUE)
colors = read.csv("/Volumes/qtran/Capper_reference_set/color.key.allRef.csv")

library(tidyr)
per_long <- gather(result_perClass[result_perClass$Dataset %in% select_dataset,], meth_class, bacc, 3:93, factor_key=TRUE)
per_long$Dataset = factor(per_long$Dataset, levels = c("35", "70_GSE", "LC", "HC"))
per_long$Seed = as.factor(per_long$Seed)
per_long$meth_class = gsub("\\.\\.", ", ", per_long$meth_class)
per_long$meth_class = gsub("\\.", " ", per_long$meth_class)

grid_labels = list('35' = "35% reference",
                   '70_GSE' = "+ all SS labels",
                   'LC' = "+ LC SS labels",
                   'HC' = "+ HC SS labels")
data_labeller <- function(variable,value){
  return(grid_labels[value])
}

fig_acc = ggplot(per_long, aes(x=meth_class, y=bacc, fill=meth_class)) + 
  geom_boxplot(outlier.shape = NA) + 
  scale_fill_manual(values = colors$color) +
  facet_grid(Dataset ~ ., labeller = data_labeller)+
  theme_bw(base_size = 14)+
  ggtitle("NN bacc perClass")+
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1),
        axis.title.x = element_blank(),
        panel.grid.minor = element_blank(), legend.position = "none")
fig_acc
ggsave(fig_acc, file = './figures/class_results/NN_bacc_holdOutTest_perclass_4groups.pdf')



