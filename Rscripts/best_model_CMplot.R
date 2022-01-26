library(ssc)
setwd("/Volumes/qtran/Semisupervised_Learning")
source("predict_setred_prob.R")
source("/Volumes/qtran/Semisupervised_Learning/supporting_functions.R")

load("./processed_data/class_results/seed20_alldata_sets.RData")
xtrain = x$xtrain
ytrain = x$ytrain
xitest = x$xitest
yitest = x$yitest
xttest = x$xttest
yttest = x$yttest

rm(x)
color_key = read.csv("/Volumes/qtran/Capper_reference_set/color.key.allRef.csv")

load("best.setred.svm.RData")
###Set up learner parameters 
learner <- e1071::svm
learner.pars <- list(type = "C-classification", kernel="radial", 
                     probability = TRUE, scale = TRUE)
pred <- function(m, x){
  r <- predict(m, x, probability = TRUE)
  prob <- attr(r, "probabilities")
  return(prob)
}
ipred = get_inductive_pred(m.setred, test_x=xitest, test_y=yitest, train_x=xtrain)
transpred = get_trans_pred(m.setred, train_x = xtrain, trans_y=yttest, tra.na.idx)

get_perm_metric = function(iLabel, predictLabel){
  #colnames(predictLabel) = gsub("\\.\\.", ", ", colnames(predictLabel))
  #colnames(predictLabel) = gsub("\\.", " ", colnames(predictLabel))
  plt = cbind(predictLabel, as.data.frame(iLabel))
  plt$predictLabel = as.character(plt$predictLabel)
  #plt$iLabel = gsub("\\.\\.", ", ", plt$iLabel)
  #plt$iLabel = gsub("\\.", " ", plt$iLabel)
  print(head(plt))
  eval = cvms::evaluate(
    data=plt,
    target_col = "iLabel",
    prediction_cols = "predictLabel",
    type="multinomial"
  )
  return (eval)
}

plot_cm = function(eval, filename, col){
  g = plot_confusion_matrix(eval, add_counts = TRUE, add_row_percentages = FALSE, 
                            palette = "Reds",tile_border_color = "gray", 
                            add_col_percentages = FALSE, add_arrows = FALSE, add_normalized = FALSE,
                            add_zero_shading = FALSE, darkness = 1,
                            place_x_axis_above = FALSE, rotate_y_text = FALSE, font_normalized = font(size =2))+
    ggplot2::labs(x = "Reference", y = "Prediction")+
    ggplot2::theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1, size=7.5, color = col$color),
                   axis.text.y = element_text(vjust = 0.5, hjust=1, size=7.5, color = col$color), 
                   legend.position = "bottom")
  
  ggsave(g, file = filename)
}

eval = get_perm_metric(yitest, ipred)

save(eval, file = "./processed_data/class_results/seed20_all/eval_seed20_itest.RData")

#color_key = read.csv("/Volumes/qtran/Capper_reference_set/color.key.allRef.csv")

plot_cm(eval, filename = "./figures/class_results/SETREDSVM_seed20_ind_testing_CM.pdf", color_key)
