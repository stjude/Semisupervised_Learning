library(dplyr)
seed1 = read.csv('/Volumes/qtran/Semisupervised_Learning/python/output/seed1_pseudo_labels_family_summary.csv')
seed1 = seed1[,-1]
diff = NULL
for(i in 1:nrow(seed1)){
  result = seed1[i+1, 3:ncol(seed1)] - seed1[i,3:ncol(seed1)]
  result$Dataset = paste0(seed1$Dataset[i+1],"-", seed1$Dataset[i])
  diff = rbind(diff, result)
  #print (result)
}
dfcount = data.frame(matrix(NA,    # Create empty data frame
                            nrow = 2,
                            ncol = 2))
colnames(dfcount) = c("LC", "HC")
rownames(dfcount) = c("LF", "HF")

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

result35 = get_HL_class(seed1$Dataset=="35", 10)
LF_fam = result35$LF
HF_fam = result35$HF

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
pseudo35 = seed1[seed1$Dataset=="70", 3:ncol(seed1)]- seed1[seed1$Dataset == "35", 3:ncol(seed1)]

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
ref_pseudo_count = read.csv('/Volumes/qtran/Semisupervised_Learning/python/output/pseudo_labels_HiLoFreq_Conf_family.csv', header=TRUE)
ref_pseudo_count$Freq_Conf = factor(ref_pseudo_count$Freq_Conf, 
                                     levels = c( "LF-LC", "HF-LC","LF-HC", "HF-HC", "LF", "HF"))
ref_pseudo_count$Frequency = factor(ref_pseudo_count$Frequency, levels=c("Low", "High"))
ref_pseudo_count$Confidence = factor(ref_pseudo_count$Confidence, levels=c("Low", "High"))

ref_pseudo_count$Dataset = factor(ref_pseudo_count$Dataset, levels=c("35", "35_GSE", "70_GSE","70HC_GSE", "70", "70_gseHC",  
                                                                     "35_gseHC", "70HC","HC"))
prop_fig = ggplot(ref_pseudo_count, aes(fill=Freq_Conf, x=Dataset, y = Proportion)) +
  geom_bar(position="stack", stat="identity" ) +
  #scale_fill_brewer(palette = "Set3")+
  ylab("Proportion")+
  geom_text(aes(y = label_y, label = round(Proportion, digits=2)), vjust = 1.5, colour = "black", size=8)+
  scale_fill_manual(values = c("#BEBADA", "#FFFFB3",  "#80B1D3", "#8DD3C7", "#FDB462",  "#FB8072"  ))+
  scale_y_continuous(breaks = seq(0, 1, by = .25)) +
  scale_x_discrete(position="top")+
  theme_bw(base_size = 22, base_rect_size = 2, base_line_size = 1.8)+
  theme(axis.text.x = element_blank(),
        axis.title.x = element_blank(),
        axis.title.y = element_text(size = 25, face="bold"),
        axis.text.y = element_text(color = "black", size = 25, face="bold"),
        #panel.grid.minor = element_blank(),
        #panel.grid.major = element_blank(),
        legend.text=element_text(size=25),
        legend.position = "none")
prop_fig
ggsave(prop_fig, file="./figures/proportion3_pseudo_labels_family.pdf")

count_fig = ggplot(ref_pseudo_count, aes(fill=Freq_Conf, x=Dataset, y = Count)) +
  geom_bar(position="stack", stat="identity" ) +
  #scale_fill_brewer(palette = "Set3")+
  ylab("Count") + 
  scale_fill_manual(values = c("#BEBADA", "#FFFFB3",  "#80B1D3", "#8DD3C7", "#FDB462",  "#FB8072"  ))+
  geom_text(aes(y = label_y1, label = Count), vjust = 1.5, colour = "black", size=8)+
  scale_y_continuous(breaks = seq(0, 3500, by = 500), position = "right") +
  scale_x_discrete(position="top")+
  theme_bw(base_size = 22, base_rect_size = 2, base_line_size = 1.8)+
  theme(axis.text.x = element_blank(),
        axis.text.y = element_text(color = "black", size=25, face = "bold"),
        axis.title.x = element_blank(),
        axis.title.y = element_text(size = 25, face="bold"),
        panel.grid.minor = element_blank(),
        panel.grid.major = element_blank(),
        legend.title = element_blank(),
        legend.text=element_text(size=25, face="bold"),
        legend.position = c(0.80, 0.80))
count_fig
ggsave(count_fig, file="./figures/count2_pseudo_labels_family.pdf")


count_class = ggplot(ref_pseudo_count, aes(fill=Freq_Conf, x=Dataset, y = Num_groups)) +
  geom_bar(position="stack", stat="identity" ) +
  #scale_fill_brewer(palette = "Set3")+
  ylab("Count") + 
  scale_fill_manual(values = c("#BEBADA", "#FFFFB3",  "#80B1D3", "#8DD3C7", "#FDB462",  "#FB8072"  ))+
  geom_text(aes(y = label_y2, label = Num_groups), vjust = 1.5, colour = "black", size=8)+
  scale_y_discrete(breaks = seq(0, 80, by = 5), position = "right") +
  scale_x_discrete(position="top")+
  theme_bw(base_size = 22, base_rect_size = 2, base_line_size = 1.8)+
  theme(axis.text.x = element_blank(),
        axis.text.y = element_text(color = "black", size=25, face = "bold"),
        axis.title.x = element_blank(),
        axis.title.y = element_text(size = 25, face="bold"),
        panel.grid.minor = element_blank(),
        panel.grid.major = element_blank(),
        legend.title = element_blank(),
        legend.text=element_text(size=25, face="bold"),
        legend.position = c(0.80, 0.82))
count_class
ggsave(count_fig, file="./figures/count2_pseudo_labels_family.pdf")