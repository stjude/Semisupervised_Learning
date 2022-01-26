library(data.table)
setDTthreads(threads=6)

ySSL = read.csv("./processed_data/class_results/Calibrated_Trans_scores_long.csv", header=TRUE)
seed1 = ySSL[ySSL$seed == "seed1_all",]
seed1 = seed1[,-1]
seed1 = seed1[order(seed1$Sample, -seed1$pred_score),]
max_scores = seed1[!duplicated(seed1$Sample),]
library(dplyr)
max_cal_scores <- seed1 %>% group_by(Sample) %>% slice(which.max(pred_score))

max_cal_scores$suprathreshold = ifelse(max_cal_scores$pred_score > 0.8, 1, 0)

max_cal_scores$Data_type = rep()