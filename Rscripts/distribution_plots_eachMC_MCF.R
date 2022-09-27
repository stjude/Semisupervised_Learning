load("/Volumes/qtran/Semisupervised_Learning/class_label_data_y.RData")

labels = read.csv("/Volumes/qtran/Capper_reference_set/GSE90496_MC_MCF_color_labels_key.csv", header=TRUE)


library(tidyr)

head(labels)
colors = read.csv("/Volumes/qtran/Capper_reference_set/color.key.allRef.csv")

ggplot(labels, aes(x = `meth.class`)) +
  geom_bar(fill = colors$color) +
  scale_y_continuous(breaks=seq(0, 150, by = 10))+
  theme_bw(base_size = 12) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        panel.grid.minor = element_blank(),
        panel.border = element_rect(size = 2),
        axis.title=element_text(size=16,face="bold"),
        axis.text.y = element_text(size=14))
ggsave("/Volumes/qtran/Semisupervised_Learning/figures/GSE90496_meth_class_distribution.pdf")


fcolor = read.csv("/Volumes/qtran/Capper_reference_set/colors.family.key.csv")
ggplot(labels, aes(x = `meth.family`)) +
  geom_bar(fill = fcolor$color) +
  scale_y_continuous(breaks=seq(0, 350, by = 20))+
  theme_bw(base_size = 12) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        panel.grid.minor = element_blank(),
        panel.border = element_rect(size = 2),
        axis.title=element_text(size=16,face="bold"),
        axis.text.y = element_text(size=14))
ggsave("/Volumes/qtran/Semisupervised_Learning/figures/GSE90496_meth_family_distribution.pdf")
