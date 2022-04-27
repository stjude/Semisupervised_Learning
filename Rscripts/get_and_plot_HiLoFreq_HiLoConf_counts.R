library(dplyr)
type = "family"
seed1 = read.csv(paste0("/Volumes/qtran/Semisupervised_Learning/python/output/seed1_pseudo_labels_", type, "_summary.csv"))
seed1 = seed1[,-1]

get_HL_class = function(data, threshold){
  LF = NULL
  HF = NULL
  for (i in 3:ncol(data)){
    if (data[,i] < threshold){
      LF = c(LF, colnames(data)[i])
    }
    else{
      HF = c(HF, colnames(data)[i])
    }
  }
  return (list(LF = LF, HF = HF))
}

get_num_groups = function(data, LF_samples, HF_samples){
  nLF = table(data[, colnames(data) %in% LF_samples] !=0)
  nHF = table(data[, colnames(data) %in% HF_samples] !=0)
  return(list(LF = nLF, HF = nHF))
}

get_num_samples = function(data, LF_samples, HF_samples){
  nLF = rowSums(data[, colnames(data) %in% LF_fam])
  nHF = rowSums(data[, colnames(data) %in% HF_fam])
  return(list(LF_samp = nLF, HF_samp = nHF))
}

result35 = get_HL_class(seed1[seed1$Dataset=="35",], 10)
LF_fam = result35$LF
HF_fam = result35$HF
samp_35 = get_num_samples(seed1[seed1$Dataset=="35", 3:ncol(seed1)], LF_fam, HF_fam)

gseLC = seed1[seed1$Dataset=="35gse",3:ncol(seed1)] - seed1[seed1$Dataset=="35gseHC", 3:ncol(seed1)]
gseHC = seed1[seed1$Dataset=="35gseHC", 3:ncol(seed1)] - seed1[seed1$Dataset=="35", 3:ncol(seed1)]
gse = seed1[seed1$Dataset=="35gse",3:ncol(seed1)] - seed1[seed1$Dataset=="35", 3:ncol(seed1)]

gse_num_Freq_HC = get_num_groups(gseHC, LF_fam, HF_fam)
gse_num_Freq_LC = get_num_groups(gseLC, LF_fam, HF_fam)
gse_samp_HC = get_num_samples(gseHC, LF_fam, HF_fam)
gse_samp_LC = get_num_samples(gseLC, LF_fam, HF_fam)

LF_35 = rowSums(seed1[1, colnames(seed1) %in% LF_fam])
HF_35 = rowSums(seed1[1, colnames(seed1) %in% HF_fam])

gseLC_LF = gseLC[, colnames(gseLC) %in% LF_fam]
gseLC_HF = gseLC[, colnames(gseLC) %in% HF_fam]
gseHC_LF = gseHC[, colnames(gseHC) %in% LF_fam]
gseHC_HF = gseHC[, colnames(gseHC) %in% HF_fam]

pseudo70HC = seed1[seed1$Dataset=="70HC", 3:ncol(seed1)] - seed1[seed1$Dataset == "35", 3:ncol(seed1)]
pseudo70LC = seed1[seed1$Dataset=="70", 3:ncol(seed1)] - seed1[seed1$Dataset=="70HC", 3:ncol(seed1)]
pseudo70 = seed1[seed1$Dataset=="70", 3:ncol(seed1)]- seed1[seed1$Dataset == "35", 3:ncol(seed1)]
p70_num_groups = get_num_groups(pseudo70, LF_fam, HF_fam)

pseudo35_sample_LC = get_num_samples(pseudo70LC, LF_fam, HF_fam)
pseudo35_sample_HC = get_num_samples(pseudo70HC, LF_fam, HF_fam)
pseudo35_group_LC = get_num_groups(pseudo70LC, LF_fam, HF_fam)
pseudo35_group_HC = get_num_groups(pseudo70HC, LF_fam, HF_fam)

pseudoLF_LC_fam = rowSums(pseudo70LC[, colnames(pseudo70LC) %in% LF_fam])
pseudoHF_LC_fam = rowSums(pseudo70LC[, colnames(pseudo70LC) %in% HF_fam])

LF_70 = rowSums(pseudo70[, colnames(pseudo70) %in% LF_fam])
HF_70 = rowSums(pseudo70[, colnames(pseudo70) %in% HF_fam])

p70LC_LF = pseudo70LC[, colnames(pseudo70LC) %in% LF_fam]
p70LC_HF = pseudo70LC[, colnames(pseudo70LC) %in% HF_fam]
p70HC_LF = pseudo70HC[, colnames(pseudo70HC) %in% LF_fam]
p70HC_HF = pseudo70HC[, colnames(pseudo70HC) %in% HF_fam]
######
library(ggplot2)
ref_pseudo_count = read.csv(paste0("/Volumes/qtran/Semisupervised_Learning/python/output/pseudo_labels_HiLoFreq_Conf_", type, ".csv"), header=TRUE)
#ref_pseudo_count$Freq_Conf = factor(ref_pseudo_count$Freq_Conf, 
#                                    levels = c( "LF-LC", "LF-HC", "LF","HF-LC", "HF-HC","HF"))
ref_pseudo_count = ref_pseudo_count[ref_pseudo_count$Dataset %in% c("35", "70_GSE","LC", "HC"),]
ref_pseudo_count$Freq_Conf = factor(ref_pseudo_count$Freq_Conf, 
                                     levels = c( "LF-LC", "HF-LC","LF-HC", "HF-HC", "LF", "HF"))
ref_pseudo_count$Frequency = factor(ref_pseudo_count$Frequency, levels=c("Low", "High"))
ref_pseudo_count$Confidence = factor(ref_pseudo_count$Confidence, levels=c("Low", "High"))

#ref_pseudo_count$Dataset = factor(ref_pseudo_count$Dataset, levels=c("35", "35_GSE", "70_GSE","70HC_GSE", "70", "70_gseHC",  
 #                                                                   "35_gseHC", "70HC","HC"))
ref_pseudo_count$Dataset = factor(ref_pseudo_count$Dataset, levels=c("35",  "70","LC", "HC"))
prop_fig = ggplot(ref_pseudo_count, aes(fill=Freq_Conf, x=Dataset, y = Proportion)) +
  geom_bar(position="stack", stat="identity", group="Frequency" ) +
  #scale_fill_brewer(palette = "Set3")+
  ylab("Proportion")+
  geom_text(aes(y = label_y, label = round(Proportion, digits=2)), vjust = 1.3, colour = "black", size=15)+
  scale_fill_manual(values = c("#BEBADA", "#FFFFB3",   "#FDB462","#80B1D3", "#8DD3C7",  "#FB8072"))+
  #scale_fill_manual(values = c("#BEBADA", "#FFFFB3",  "#80B1D3", "#8DD3C7", "#FDB462",  "#FB8072"  ))+
  scale_y_continuous(breaks = seq(0, 1, by = .25)) +
  scale_x_discrete(position="top")+
  theme_bw(base_size = 30, base_rect_size = 3, base_line_size = 1.8)+
  theme(axis.text.x = element_blank(),
        axis.title.x = element_blank(),
        axis.title.y = element_text(size = 40, face="bold"),
        axis.text.y = element_text(color = "black", size = 40, face="bold"),
        panel.grid.minor = element_blank(),
        panel.grid.major = element_blank(),
        legend.text=element_text(size=25),
        legend.position = "none")
prop_fig
ggsave(prop_fig, file=paste0("/Volumes/qtran/Semisupervised_Learning/figures/proportion3_pseudo_labels_4groups_", type, "_v3.pdf"))

count_fig = ggplot(ref_pseudo_count, aes(fill=Freq_Conf, x=Dataset, y = Count)) +
  geom_bar(position="stack", stat="identity" ) +
  #scale_fill_brewer(palette = "Set3")+
  ylab("Count") + 
  scale_fill_manual(values = c("#BEBADA", "#FFFFB3",   "#FDB462","#80B1D3", "#8DD3C7",  "#FB8072"))+
  #scale_fill_manual(values = c("#BEBADA", "#FFFFB3",  "#80B1D3", "#8DD3C7", "#FDB462",  "#FB8072"  ))+
  geom_text(aes(y = label_y1, label = Count), vjust = 1.1, colour = "black", size=15, face="bold")+
  scale_y_continuous(breaks = seq(0, 3500, by = 500), position = "right") +
  scale_x_discrete(position="top")+
  theme_bw(base_size = 30, base_rect_size = 3, base_line_size = 1.8)+
  theme(axis.text.x = element_blank(),
        axis.text.y = element_text(color = "black", size=40, face = "bold"),
        axis.title.x = element_blank(),
        axis.title.y = element_text(size = 40, face="bold"),
        panel.grid.minor = element_blank(),
        panel.grid.major = element_blank(),
        legend.title = element_blank(),
        legend.text=element_text(size=35, face="bold"),
        legend.position = c(0.15, 0.80))
count_fig
ggsave(count_fig, file=paste0("/Volumes/qtran/Semisupervised_Learning/figures/count2_pseudo_labels_4groups_", type, "_reorder_Freq_v3.pdf"))


######
ref_pseudo_count = read.csv(paste0("./python/output/pseudo_labels_HiLoFreq_Conf_", type, ".csv"), header=TRUE)
#ref_pseudo_count = ref_pseudo_count[!(ref_pseudo_count$Freq_Conf %in% c("LF", "HF")), ]
ref_pseudo_count$Freq_Conf = factor(ref_pseudo_count$Freq_Conf, 
                                    levels = c( "LF-LC", "LF-HC", "LF","HF-LC", "HF-HC","HF"))
ref_pseudo_count$Frequency = factor(ref_pseudo_count$Frequency, levels=c("Low", "High"))

ref_pseudo_count$Confidence = factor(ref_pseudo_count$Confidence, levels=c("Low", "High"))

ref_pseudo_count$Dataset = factor(ref_pseudo_count$Dataset, levels=c("35", "35_GSE", "70_GSE","70HC_GSE", "70", "70_gseHC",  
                                                                     "35_gseHC", "70HC","HC"))
count_class = ggplot(ref_pseudo_count, aes(x=Dataset, y = Added_groups, fill=Freq_Conf)) +
  geom_bar(position="stack", stat="identity" ) +
  #scale_fill_brewer(palette = "Set3")+
  ylab("Number of methylation groups") + 
  scale_fill_manual(values = c("#BEBADA",   "#80B1D3", "#FDB462", "#FFFFB3","#8DD3C7",  "#FB8072"))+
  geom_text(aes(y = Label_added_groups, label = Added_groups), vjust = 1.5, colour = "black", size=8)+
  #scale_y_discrete(breaks = seq(0, 100, by = 10), position = "left") +
  scale_x_discrete(position="top")+
  theme_bw(base_size = 22, base_rect_size = 2, base_line_size = 1.8)+
  theme(#axis.text.x = element_blank(),
        axis.text.y = element_text(color = "black", size=25, face = "bold"),
        axis.title.x = element_blank(),
        axis.title.y = element_text(size = 25, face="bold"),
        panel.grid.minor = element_blank(),
        panel.grid.major = element_blank(),
        legend.title = element_blank(),
        legend.text=element_text(size=25, face="bold"),
        legend.position = c(0.80, 0.83))
count_class
ggsave(count_fig, ile=paste0("./figures/count_family_groups_pseudo_labels_", type, ".pdf"))

ggplot(ref_pseudo_count, aes(x=Dataset, y = Added_groups, group = Freq_Conf, color=Freq_Conf)) +
  geom_line()+
  geom_point()
  #scale_fill_brewer(palette = "Set3")+
  ylab("Number of methylation groups") + 
  scale_fill_manual(values = c("#BEBADA", "#FFFFB3",  "#80B1D3", "#8DD3C7", "#FDB462",  "#FB8072"  ))+
  geom_text(aes(y = Label_added_groups, label = Added_groups), vjust = 1.5, colour = "black", size=8)+
  #scale_y_discrete(breaks = seq(0, 100, by = 10), position = "left") +
  scale_x_discrete(position="top")+
  theme_bw(base_size = 22, base_rect_size = 2, base_line_size = 1.8)+
  theme(#axis.text.x = element_blank(),
    axis.text.y = element_text(color = "black", size=25, face = "bold"),
    axis.title.x = element_blank(),
    axis.title.y = element_text(size = 25, face="bold"),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_blank(),
    legend.title = element_blank(),
    legend.text=element_text(size=25, face="bold"),
    legend.position = c(0.80, 0.83))