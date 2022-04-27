#base_dir = "/research/rgs01/home/clusterHome/qtran/Semisupervised_Learning/processed_data/family_results"
base_dir = "/Volumes/qtran/Semisupervised_Learning_big_private_data/processed_data/family_results"
seeds = c('seed1_all', 'seed2_all', 'seed20_all', 'seed40_all', 'seed80_all', 'seed160_all', 'seed320_all')
model = c('setred_OneNN', 'snnrce_OneNN', 'selfTraining_OneNN')
paths=file.path(base_dir, seeds)
#model= 'setred_OneNN'
#seeds= 'seed1_all'
num = nchar(base_dir) 
#######
#@data:
#@seed: the seed folder
#@filename: the absolute path filname
#@type: c("Inductive_metrics_", "Transductive_metrics_")
#######
attach_info = function(data, seed, filename, type){
  start = num + nchar(seed) + nchar(type) + 3
  print(start)
  data$seed = seed
  trainer = substring(filename, start, nchar(filename)-4)
  print(trainer)
  print(paste0("Trainer = ", trainer))
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

transpose_to_long= function(data){
  colnames(data)[1] = "Sample"
  ilong = tidyr::gather(data, pred_family, pred_score, 2:76, factor_key=FALSE)
  ilong$pred_family = gsub("\\.\\.", ", ", ilong$pred_family)
  ilong$pred_family = gsub(".", " ", ilong$pred_family, fixed=TRUE)
  ilong$match = ifelse(ilong$truth_label == ilong$pred_family, 1, 0)
  
  return(ilong)
}
###############
for (m in model){
  ind_data = NULL
  trans_data = NULL
  ind_score = NULL
  trans_score = NULL
  max_iscore = NULL
  max_tscore = NULL
  for (s in seeds){
    p = file.path(base_dir, s)
    #indfiles <- list.files(path = p, pattern='Inductive_metrics_*', full.names=TRUE, recursive=FALSE)
    #transfiles <- list.files(path = p, pattern='Transductive_metrics_*', full.names=TRUE, recursive=FALSE)
    ind_score_file <- list.files(path = p, pattern=paste0("Inductive_scores_", m), full.names=TRUE, recursive=FALSE )
    trans_score_file <- list.files(path = p, pattern=paste0('Transductive_scores_', m), full.names=TRUE, recursive=FALSE )
    max_ind_score_file <- list.files(path = p, pattern=paste0('Inductive_MaxScores_', m), full.names=TRUE, recursive=FALSE )
    max_trans_score_file <- list.files(path = p, pattern=paste0('Transductive_MaxScores_', m), full.names=TRUE, recursive=FALSE )
  
    iscore = read.csv(ind_score_file, header=TRUE) # load file
    max_indscore <- read.csv(max_ind_score_file, header=TRUE)
    iscore_new = attach_info(iscore, seed = s, filename=ind_score_file, type = "Inductive_scores_")
    iscore_new$truth_label = max_indscore$truth_label[match(iscore_new$X, max_indscore$Sample)]
    ind_score = rbind(ind_score, iscore_new)
    print("Got Inductive scores")
      
    transcore = read.csv(trans_score_file, header=TRUE) # load file
    max_transcore <- read.csv(max_trans_score_file, header=TRUE)
    transcore_new = attach_info(transcore, seed = s, filename=trans_score_file, type = "Transductive_scores_")
    transcore_new$truth_label = max_transcore$truth_label[match(transcore_new$X, max_transcore$Sample)]
    trans_score = rbind(trans_score, transcore_new)
    print("Got Transductive scores")
    
    print("HERE")
    p = file.path(base_dir, s)
    max_ind_score_file <- list.files(path = p, pattern=paste0('Inductive_MaxScores_',m), full.names=TRUE, recursive=FALSE )
    max_trans_score_file <- list.files(path = p, pattern=paste0('Transductive_MaxScores_',m), full.names=TRUE, recursive=FALSE )
    
    itemp = read.csv(max_ind_score_file, header=TRUE)
    itemp$seed = s
    max_iscore = rbind(max_iscore, itemp)
    
    ttemp = read.csv(max_trans_score_file, header=TRUE)
    ttemp$seed = s
    max_tscore = rbind(max_tscore, itemp)
  }
  
  colnames(ind_score)[1] = "Sample"
  colnames(trans_score)[1] = "Sample"

  ind_score_long = transpose_to_long(ind_score)
  trans_score_long = transpose_to_long(trans_score)
  
  write.csv(ind_score_long, file = paste0(base_dir, "/All_MCF_", m, "_Inductive_scores_long.csv"), row.names=FALSE)
  write.csv(trans_score_long, file = paste0(base_dir, "/All_MCF_", m, "_Transductive_scores_long.csv"), row.names=FALSE)
  
  write.csv(ind_score, file = paste0(base_dir, "/All_MCF_", m, "_Inductive_scores_wide.csv"), row.names=FALSE)
  write.csv(trans_score, file = paste0(base_dir, "/All_MCF_", m,"_Transductive_scores_wide.csv"), row.names=FALSE)
  
  write.csv(max_iscore, file = paste0(base_dir, "/All_MCF_Inductive_MaxScores_", m, ".csv"), row.names=FALSE)
  write.csv(max_tscore, file = paste0(base_dir, "/All_MCF_Transductive_MaxScores_", m, ".csv"), row.names=FALSE)
}
