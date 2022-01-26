base_dir = "/research/rgs01/home/clusterHome/qtran/Semisupervised_Learning/processed_data/class_results"
bdir = "/Volumes/qtran/Semisupervised_Learning/processed_data/class_results"
seeds = c('seed1_all', 'seed2_all', 'seed20_all', 'seed40_all', 'seed80_all', 'seed160_all', 'seed320_all')
paths=file.path(base_dir, seeds)

num = nchar(base_dir) 
#######
#@data:
#@seed: the seed folder
#@filename: the absolute path filname
#@type: c("Inductive_metrics_", "Transductive_metrics_")
#######
attach_info = function(data, seed, filename, type){
  start = num + nchar(seed) + nchar(type) + 3
  data$seed = seed
  trainer = substring(filename, start, nchar(filename)-4)
  #print(paste0("Trainer = ", trainer))
  data$Trainer = trainer
  print(paste0("Trainer = ", trainer))
  if(grepl("OneNN", trainer) | grepl("8", trainer)){
    print(paste0("1NN = ", trainer))
    
    data$Learner = "1NN"
  } else if (grepl("9", trainer) | grepl("svm", trainer)){
    print(paste0("SVM = ", trainer))
    
    data$Learner = "SVM"
  } else if (grepl("10", trainer) | grepl("C5.0", trainer)){
    print(paste0("C5.0 = ", trainer))
    
    data$Learner = "C5.0"
  } else if (grepl("c_", trainer)){
    print(paste0("DEMO = ", trainer))
    data$Learner = "1NN/SVM/C5.0"
  }else{
    print("Doesn't match!!!")
  }
  return(data)
}

combine_files = function(long_data, t, seed){
  if (seed == 'seed1_all'){
    print( seed)
    data = rbind(long_data, t)
  }
  else{
    data = rbind(long_data, t)
  }
  return(data)
}

###############

ind_data = NULL
trans_data = NULL
ind_score = NULL
trans_score = NULL
for (s in seeds){
  p = file.path(base_dir, s)
  indfiles <- list.files(path = p, pattern='Inductive_metrics_*', full.names=TRUE, recursive=FALSE)
  transfiles <- list.files(path = p, pattern='Transductive_metrics_*', full.names=TRUE, recursive=FALSE)
  ind_score_file <- list.files(path = p, pattern='Inductive_scores_setred_*', full.names=TRUE, recursive=FALSE )
  trans_score_file <- list.files(path = p, pattern='Transductive_scores_setred_*', full.names=TRUE, recursive=FALSE )
  max_ind_score_file <- list.files(path = p, pattern='Inductive_MaxScores_setred_*', full.names=TRUE, recursive=FALSE )
  max_trans_score_file <- list.files(path = p, pattern='Transductive_MaxScores_setred_*', full.names=TRUE, recursive=FALSE )
  
  print(indfiles)
  for (i in indfiles){
    ind = read.csv(i, header=TRUE) # load file
    ind_new = attach_info(ind, seed = s, filename=i, type = "Inductive_metrics_")
    ind_data = rbind(ind_data, ind_new)
  }
  
  for (j in transfiles){
    trans = read.csv(j, header=TRUE) # load file
    trans_new = attach_info(trans, seed = s, filename=j, type = "Transductive_metrics_")
    trans_data = rbind(trans_data, trans_new)
  }
  
  iscore = read.csv(ind_score_file, header=TRUE) # load file
  max_indscore <- read.csv(max_ind_score_file, header=TRUE)
  iscore_new = attach_info(iscore, seed = s, filename=ind_score_file, type = "Inductive_scores_")
  iscore_new$truth_label = max_indscore$truth_label[match(iscore_new$X, max_indscore$Sample)]
  ind_score = rbind(ind_score, iscore_new)
  
    
  transcore = read.csv(trans_score_file, header=TRUE) # load file
  max_transcore <- read.csv(max_trans_score_file, header=TRUE)
  transcore_new = attach_info(transcore, seed = s, filename=trans_score_file, type = "Transductive_scores_")
  transcore_new$truth_label = max_transcore$truth_label[match(transcore_new$X, max_transcore$Sample)]
  trans_score = rbind(trans_score, transcore_new)
  
}

ind_data$TrainFunction = ifelse(grepl("tri", ind_data$Trainer), "TRITRAIN", 
                          ifelse(grepl("snnrce", ind_data$Trainer), "SNNRCE", 
                                 ifelse(grepl("democratic", ind_data$Trainer), "DEMO", 
                                        ifelse(grepl("setred", ind_data$Trainer), "SETRED", "SELFT"))))

trans_data$TrainFunction = ifelse(grepl("tri", trans_data$Trainer), "TRITRAIN", 
                          ifelse(grepl("snnrce", trans_data$Trainer), "SNNRCE", 
                                 ifelse(grepl("democratic", trans_data$Trainer), "DEMO", 
                                        ifelse(grepl("setred", trans_data$Trainer), "SETRED", "SELFT"))))
call_type = "Class"
colnames(ind_data)[1]  = call_type
colnames(trans_data)[1]  = call_type

ind_data$Class = gsub("Class: ", "", ind_data$Class)
trans_data$Class = gsub("Class: ", "", trans_data$Class)

write.csv(ind_data, file = paste0(base_dir, "/All_Inductive_metrics.csv"), row.names=FALSE)
write.csv(trans_data, file = paste0(base_dir, "/All_Transductive_metrics.csv"), row.names=FALSE)

colnames(ind_score)[1] = "Sample"
colnames(trans_score)[1] = "Sample"
library(tidyr)
ind_score_long <- gather(ind_score, pred_label, pred_score, 2:92, factor_key=FALSE)
trans_score_long <- gather(trans_score, pred_label, pred_score, 2:92, factor_key=FALSE)

ind_score_long$pred_label = gsub("\\.\\.", ", ", ind_score_long$pred_label)
ind_score_long$pred_label = gsub(".", " ", ind_score_long$pred_label, fixed=TRUE)
ind_score_long$match = ifelse(ind_score_long$truth_label == ind_score_long$pred_label, 1, 0)

trans_score_long$pred_label = gsub("\\.\\.", ", ", trans_score_long$pred_label)
trans_score_long$pred_label = gsub(".", " ", trans_score_long$pred_label, fixed=TRUE)
trans_score_long$match = ifelse(trans_score_long$truth_label == trans_score_long$pred_label, 1, 0)

write.csv(ind_score_long, file = paste0(base_dir, "/All_SETRED_SVM_Inductive_scores_long.csv"), row.names=FALSE)
write.csv(trans_score_long, file = paste0(base_dir, "/All_SETRED_SVM_Transductive_scores_long.csv"), row.names=FALSE)

write.csv(ind_score, file = paste0(base_dir, "/All_SETRED_SVM_Inductive_scores_wide.csv"), row.names=FALSE)
write.csv(trans_score, file = paste0(base_dir, "/All_SETRED_SVM_Transductive_scores_wide.csv"), row.names=FALSE)

max_iscore = NULL
max_tscore = NULL
for (s in seeds){
  p = file.path(base_dir, s)
  max_ind_score_file <- list.files(path = p, pattern='Inductive_MaxScores_setred_*', full.names=TRUE, recursive=FALSE )
  max_trans_score_file <- list.files(path = p, pattern='Transductive_MaxScores_setred_*', full.names=TRUE, recursive=FALSE )
  
  itemp = read.csv(max_ind_score_file, header=TRUE)
  itemp$seed = s
  max_iscore = rbind(max_iscore, itemp)
  
  ttemp = read.csv(max_trans_score_file, header=TRUE)
  ttemp$seed = s
  max_tscore = rbind(max_tscore, itemp)
}

write.csv(max_iscore, file = paste0(base_dir, "/All_Inductive_MaxScores_SETRED_SVM.csv"), row.names=FALSE)
write.csv(max_tscore, file = paste0(base_dir, "/All_Transductive_MaxScores_SETRED_SVM.csv"), row.names=FALSE)
