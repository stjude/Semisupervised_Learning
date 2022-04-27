
library(ggplot2)
library(dplyr)
base_dir = "/Volumes/qtran/Semisupervised_Learning/figures/class_results/"

ind_data = read.csv("/Volumes/qtran/Semisupervised_Learning_big_private_data/processed_data/class_results/All_Inductive_metrics.csv", header=TRUE)
ind_data$Class = as.factor(ind_data$Class)
ind_data$TrainFunction = as.factor(ind_data$TrainFunction)
ind_data$Learner = as.factor(ind_data$Learner)

trans_data = read.csv("/Volumes/qtran/Semisupervised_Learning_big_private_data/processed_data/class_results/All_Transductive_metrics.csv", header=TRUE)
trans_data$Class = as.factor(trans_data$Class)
trans_data$TrainFunction = as.factor(trans_data$TrainFunction)
trans_data$Learner = as.factor(trans_data$Learner)

#####SPECIFICITY#####
create_overall_plot = function(idata, tdata, metric = c("Balanced.Accuracy", "Specificity", "Precision", "Recall"), basedir, file ){
  if (metric == "Specificity") {
    print("HERE")
    ind = idata %>% group_by(TrainFunction, Learner, seed) %>% 
      summarise_at(vars(Specificity), list(~mean(., na.rm=TRUE)))
    trans = tdata %>% group_by(TrainFunction, Learner, seed) %>% 
      summarise_at(vars(Specificity), list(~mean(., na.rm=TRUE)))
  } else if (metric == "Balanced.Accuracy") {
    ind = idata %>% group_by(TrainFunction, Learner, seed) %>% 
      summarise_at(vars(Balanced.Accuracy), list(~mean(., na.rm=TRUE)))
    trans = tdata %>% group_by(TrainFunction, Learner, seed) %>% 
      summarise_at(vars(Balanced.Accuracy), list(~mean(., na.rm=TRUE)))
  }else if (metric == "Precision") {
    ind = idata %>% group_by(TrainFunction, Learner, seed) %>% 
      summarise_at(vars(Precision), list(~mean(., na.rm=TRUE)))
    trans = tdata %>% group_by(TrainFunction, Learner, seed) %>% 
      summarise_at(vars(Precision), list(~mean(., na.rm=TRUE)))
  }else if (metric == "Recall") {
    ind = idata %>% group_by(TrainFunction, Learner, seed) %>% 
      summarise_at(vars(Recall), list(~mean(., na.rm=TRUE)))
    trans = tdata %>% group_by(TrainFunction, Learner, seed) %>% 
      summarise_at(vars(Recall), list(~mean(., na.rm=TRUE)))
  }else{
    print("Metric is not found!")
  }
  
  ind$TestType = "Inductive"
  trans$TestType = "Transductive"
  
  acc = rbind(ind, trans)
  acc$Learner = as.factor(acc$Learner)
  acc$TrainFunction = as.factor(acc$TrainFunction)
  acc = as.data.frame(acc)
  colnames(acc)[4] = "value"
  acc_plot = ggplot(acc, aes(x = Learner , y = value, fill=Learner)) + 
    geom_boxplot(outlier.shape = 8)+
    #scale_fill_manual(values = c("#009999", "#FFBF00")) +
    facet_grid(TestType ~TrainFunction) +
    labs(x = "Learner", y = eval(metric)) +
    ylim(0.997,1) +
    #theme_minimal() +
    theme(axis.text.y   = element_text(size=20),
          axis.text.x   = element_text(size=18, angle = 30, hjust = 1),
          axis.title.y  = element_text(size=20),
          axis.title.x  = element_text(size=20),
          #panel.grid.minor.y = element_blank(),
          #panel.grid.major.x = element_blank(),
          panel.grid.major.y = element_line(color = "gray", size = 1, linetype = "dotted"),
          panel.grid.major.x = element_line(color = "gray", size = 1, linetype = "dotted"),
          panel.background = element_blank(), 
          panel.border = element_rect(color="black", size=3, fill=NA),
          strip.text.x = element_text(size = 20),
          strip.text.y = element_text(size = 20),
          legend.text = element_text(size = 18),
          legend.title = element_blank(),
          legend.position = "bottom") 
  
  ggsave(acc_plot, file = paste0(basedir, file)
         ,width = 11, height = 7, units="in")

}

create_overall_plot(ind_data, trans_data, metric="Specificity", basedir = base_dir, file = "Specificity_Overall_zoom.pdf")



#####ACCURACY#####
acc_ind = ind_data %>% group_by(TrainFunction, Learner, seed) %>% 
  summarise_at(vars(Balanced.Accuracy), list(~mean(., na.rm=TRUE)))
acc_trans = trans_data %>% group_by(TrainFunction, Learner, seed) %>% 
  summarise_at(vars(Balanced.Accuracy), list(~mean(., na.rm=TRUE)))
acc_ind$TestType = "Inductive"
acc_trans$TestType = "Transductive"

acc = rbind(acc_ind, acc_trans)
#acc = merge(acc_ind, acc_trans, by = c("TrainFunction", "Learner", "seed"))
#colnames(acc)[4:5] = c("Inductive.Accuracy", "Transductive.Accuracy")
acc$Learner = as.factor(acc$Learner)
acc$TrainFunction = as.factor(acc$TrainFunction)

acc_long_plot = ggplot(acc, aes(x = Learner , y = Balanced.Accuracy, fill=Learner)) + 
  geom_boxplot(outlier.shape = 8)+
  #scale_fill_manual(values = c("#009999", "#FFBF00")) +
  facet_grid(TestType ~TrainFunction) +
  labs(x = "Learner", y = "Accuracy") +
  ylim(0.8,1) +
  theme(axis.text.y   = element_text(size=20),
        axis.text.x   = element_text(size=18, angle = 30, hjust = 1),
        axis.title.y  = element_text(size=20),
        axis.title.x  = element_text(size=20),
        panel.grid.major.y = element_line(color = "gray", size = 1, linetype = "dotted"),
        panel.grid.major.x = element_line(color = "gray", size = 1, linetype = "dotted"),
        panel.background = element_blank(), 
        panel.border = element_rect(color="black", size=3, fill=NA),
        strip.text.x = element_text(size = 20),
        strip.text.y = element_text(size = 20),
        legend.text = element_text(size = 18),
        legend.title = element_blank(),
        legend.position = "bottom")

ggsave(acc_long_plot, file = paste0(base_dir, "Accuracy_overall_zoom.pdf")
       ,width = 11, height = 7, units="in")

###### PREC-REC PLOT FUNCTION#####
pr_plot = function(data, type=c("Inductive", "Transductive"), based_dir){
  if( type %in% c ("Inductive", "Transductive")){
    pr= data %>% group_by(TrainFunction, Learner, seed) %>% 
      summarise_at(vars(Precision, Recall), list(~mean(., na.rm=TRUE)))
  }else{
    stop("Type of testing is not correct!")
  }
  gpr = ggplot(pr, aes(x = Recall, y = Precision, fill=Learner)) + 
    geom_boxplot(outlier.shape = 8)+
    labs(title = paste(type, "Testing")) +
    theme(axis.text.y   = element_text(size=14),
          axis.text.x   = element_text(size=11),
          axis.title.y  = element_text(size=14),
          axis.title.x  = element_text(size=14),
          panel.grid.major.y = element_line(color = "gray", size = 0.5, linetype = "dotted"),
          panel.background = element_blank(), 
          panel.border = element_rect(color="black", size=1, fill=NA),
          strip.text.x = element_text(size = 14),
          legend.position = "bottom")+
    facet_grid(cols = vars(TrainFunction)) 
  
  ggsave(gpr, file = paste0(base_dir, type, "_Prec_Recall_overall.pdf"),
         width = 14, height = 6, units="in")
}

pr_plot(ind_data, type= "Inductive", base_dir)
pr_plot(trans_data, type= "Transductive", based_dir)

########
#############GET ACC average for inductive and transductive testing#####
acc_long = rbind(acc_ind, acc_trans)
ACC_table = acc_long %>% group_by(TrainFunction, Learner, TestType) %>% 
  summarise_at(vars(Balanced.Accuracy), list(~mean(., na.rm=TRUE)))
#########PREC-RECALL INDUCTIVE AND TRANSDUCTIVE COMBINED#####
pr_ind= ind_data %>% group_by(TrainFunction, Learner, seed) %>% 
  summarise_at(vars(Precision, Recall), list(~mean(., na.rm=TRUE)))
pr_ind$TestType = "Inductive"
pr_trans = trans_data %>% group_by(TrainFunction, Learner, seed) %>% 
  summarise_at(vars(Precision, Recall), list(~mean(., na.rm=TRUE)))
pr_trans$TestType = "Transductive"

pre_call = rbind(pr_ind, pr_trans)

pre_call$Learner = as.factor(pre_call$Learner)
pre_call$TrainFunction = as.factor(pre_call$TrainFunction)

library(tidyr)
pr_long <- gather(pre_call, Metrics, Values, Precision:Recall, factor_key=TRUE)

PR_table = pre_call %>% group_by(TrainFunction, Learner, TestType) %>% 
  summarise_at(vars(Precision, Recall), list(~mean(., na.rm=TRUE)))

pra_table = merge(ACC_table, PR_table, by = c("TrainFunction", "Learner", "TestType"))
write.csv(pra_table, file="./processed_data/class_results/Average_PR_Acc_byTestType.csv")

#####
pr_long_plot = ggplot(pr_long, aes(x = Learner, y = Values, fill=Learner)) + 
  geom_boxplot() + 
  facet_grid(TestType+Metrics~TrainFunction) +
  labs(y = "Proportion", x="Learner") +
  ylim(0.7,1)+
  theme(axis.text.y   = element_text(size=18),
        axis.text.x   = element_text(size=18, angle = 30, hjust =1),
        axis.title.y  = element_text(size=18),
        axis.title.x  = element_text(size=18),
        panel.grid.major.y = element_line(color = "gray", size = 1, linetype = "dotted"),
        panel.grid.major.x = element_line(color = "gray", size = 1, linetype = "dotted"),
        panel.background = element_blank(), 
        panel.border = element_rect(color="black", size=3, fill=NA),
        strip.text.x = element_text(size = 14),
        strip.text.y = element_text(size = 18),
        legend.title = element_blank(),
        legend.position = "bottom", legend.text=element_text(size=16))

ggsave(pr_long_plot, file = paste0(base_dir, "PR_Combined_Long_Overall_zoom.pdf"),
       width = 11, height = 8, units="in")
#####


####################################################3
acc_ind_class = ind_data %>% group_by(Class, TrainFunction, Learner, seed) %>% 
  summarise_at(vars(Balanced.Accuracy), funs(mean(., na.rm=TRUE)))

acc_trans_class = trans_data %>% group_by(Class, TrainFunction, Learner, seed) %>% 
  summarise_at(vars(Balanced.Accuracy), funs(mean(., na.rm=TRUE)))

######
color_key = read.csv("/Volumes/qtran/Capper_reference_set/color.key.allRef.csv")

draw_overall_acc = function(data, title, base_dir, metrics = c("Precision", "Recall", "Specificity", "Balanced.Accuracy"),type = c("Inductive", "Transductive")){
   if (metrics == "Precision"){
     sum_data = data %>% group_by(TrainFunction, Learner, seed) %>% 
       summarise_at(vars(Precision), list(~mean(., na.rm=TRUE)))
     g = ggplot(sum_data, aes(x=Learner, y=Precision, fill=Learner)) +
        facet_wrap(vars(TrainFunction))+
        geom_boxplot() +
        geom_jitter(shape=16, position=position_jitter(0.2)) +
        labs(title = title, y = metrics) +
        theme_bw(base_size = 16) +
        theme(legend.position = "none")
   }else if (metrics == "Recall"){
     sum_data = data %>% group_by(TrainFunction, Learner, seed) %>% 
       summarise_at(vars(Recall), list(~mean(., na.rm=TRUE)))
     g = ggplot(sum_data, aes(x=Learner, y=Recall, fill=Learner)) +
       facet_wrap(vars(TrainFunction))+
       geom_boxplot() +
       geom_jitter(shape=16, position=position_jitter(0.2)) +
       labs(title = title, y = metrics) +
       theme_bw(base_size = 16) +
       theme(legend.position = "none")
   }else if (metrics == "Specificity"){
     sum_data = ind_data %>% group_by(TrainFunction, Learner, seed) %>% 
       summarise_at(vars(Specificity), list(~mean(., na.rm=TRUE)))
     print(head(sum_data))
     g = ggplot(sum_data, aes(x=Learner, y=Specificity, fill=Learner)) +
       facet_wrap(vars(TrainFunction))+ ylim(0.99, 1) +
       geom_boxplot() +
       geom_jitter(shape=16, position=position_jitter(0.2)) +
       labs(title = title, y = metrics) +
       theme_bw(base_size = 16) +
       theme(legend.position = "none")
   }else {
     sum_data = data %>% group_by(TrainFunction, Learner, seed) %>% 
       summarise_at(vars(Balanced.Accuracy), list(~mean(., na.rm=TRUE)))
     g = ggplot(sum_data, aes(x=TrainFunction, y=Balanced.Accuracy, fill=Learner)) +
       #facet_wrap(vars(TrainFunction))+
       geom_boxplot() +
       #geom_jitter(shape=16, position=position_jitter(0.2)) +
       labs(title = title, y = metrics) +
       theme_classic(base_size = 16) +
       theme(legend.position = "bottom")
   }
  ggsave(g, file = paste0(base_dir, metrics, "_", type, "_overall.pdf"),
         width = 13, height = 9, units="in")
}


draw_acc_by_class = function(data, base_dir, col, trainFunction, type = c("Inductive", "Transductive")){
  if (!(is.null(trainFunction))){
    d = data[data$TrainFunction==trainFunction,]
  }
  else{
    d = data
    trainFunction = "All"
  }
  if (length(unique(d$Learner)) > 1){
    print(length(unique(d$Learner)))
    text_size = 5
  }else{
    text_size = 9
  }
  
  g =ggplot(d, aes(x=Class, y=Balanced.Accuracy, fill = Class)) +
      facet_wrap(vars(Learner)) +
      scale_fill_manual(values = col)+ ylim(0,1) +
      geom_boxplot() +
      geom_jitter(shape=16, position=position_jitter(0.2)) +
      labs(title = paste(trainFunction, "-", type, "Accuracy by Class")) +
      theme_bw(base_size = 14) +
      theme(legend.position = "none", axis.text.x= element_text(size = text_size, angle = 90, hjust = 1, vjust = 0.5))

  ggsave(g, file = paste0(base_dir, trainFunction, "_", type, "_accuracy_byClass.pdf"), 
         width = 17, height = 7, units="in")
}


m = c("Precision", "Recall", "Balanced.Accuracy")
m = "Specificity"
for (i in m){
  draw_overall_acc(ind_data, title = paste(i, "for Inductive Testing (30% hold out)"),
                 metrics=i, type = "Inductive", base_dir = base_dir)
  draw_overall_acc(trans_data, title = paste(i, "for Transductive Testing (50% training data)"), 
                   metrics = i, type = "Transductive", base_dir = base_dir)
}

################GET ACC of BEST MODELS#########
best_trainer = c("SETRED", "DEMO", "SELFT")
best_learner = c("SVM", "1NN/SVM/C5.0", "1NN")

sub_ind_acc = acc_ind_class[(acc_ind_class$TrainFunction %in% best_trainer
                             & acc_ind_class$Learner %in% best_learner), ]
sub_ind_acc = sub_ind_acc[!(sub_ind_acc$TrainFunction == "SELFT"
                          & sub_ind_acc$Learner == "SVM"), ]
sub_ind_acc = sub_ind_acc[!(sub_ind_acc$TrainFunction == "SETRED"
                           & sub_ind_acc$Learner == "1NN"), ]
sub_ind_acc$TestType = "Inductive"
sub_trans_acc = acc_trans_class[(acc_trans_class$TrainFunction %in% best_trainer
                             & acc_trans_class$Learner %in% best_learner), ]
sub_trans_acc = sub_trans_acc[!(sub_trans_acc$TrainFunction == "SELFT"
                            & sub_trans_acc$Learner == "SVM"), ]
sub_trans_acc = sub_trans_acc[!(sub_trans_acc$TrainFunction == "SETRED"
                            & sub_trans_acc$Learner == "1NN"), ]
sub_trans_acc$TestType = "Transductive"

sub_acc = rbind(sub_ind_acc, sub_trans_acc)

######DRAW BEST MODEL ACC BY CLASS########
tf.labs <- c("DEMO", "SELFT (1NN)", "SETRED (SVM)")
names(tf.labs) <- c("DEMO", "SELFT", "SETRED")
sub_acc$Class = as.factor(sub_acc$Class)
g1 =ggplot(sub_acc, aes(x=Class, y=Balanced.Accuracy, fill = Class)) +
  facet_grid(rows =   TrainFunction~TestType,
             labeller = labeller(TrainFunction = tf.labs)) +
  scale_fill_manual(values = color_key$color)+
  geom_boxplot() + ylim(0,1) +
  #geom_jitter(shape=16, position=position_jitter(0.2)) +
  #labs(title = paste(TrainFunction, "-", type, "Accuracy by Class")) +
  theme_bw(base_size = 14) +
  theme(legend.position = "none", 
        axis.text.x= element_text(size = 7, angle = 90, hjust = 1, vjust = 0.5))

ggsave(g1, file = paste0(base_dir, "Best_Models_Accuracy_byClass.pdf"), 
       width = 19, height = 10, units="in")


###################GET PRECISION RECALL SPECIFICITY of BEST MODELS##########
pr_ind_class= ind_data %>% group_by(Class,TrainFunction, Learner, seed) %>% 
  summarise_at(vars(Precision, Recall, Specificity), list(~mean(., na.rm=TRUE)))
pr_ind_class$TestType = "Inductive"
pr_trans_class = trans_data %>% group_by(Class,TrainFunction, Learner, seed) %>% 
  summarise_at(vars(Precision, Recall), list(~mean(., na.rm=TRUE)))
pr_trans_class$TestType = "Transductive"

pre_call_class = rbind(pr_ind_class, pr_trans_class)

pre_call_class$Learner = as.factor(pre_call_class$Learner)
pre_call_class$TrainFunction = as.factor(pre_call_class$TrainFunction)



best_pr = pre_call_class[(pre_call_class$TrainFunction %in% best_trainer
                         & pre_call_class$Learner %in% best_learner), ]
best_pr = best_pr[!(best_pr$TrainFunction == "SELFT"
                            & best_pr$Learner == "SVM"), ]
best_pr = best_pr[!(best_pr$TrainFunction == "SETRED"
                            & best_pr$Learner == "1NN"), ]
g2 =ggplot(best_pr, aes(x=Class, y=Precision, fill = Class)) +
  facet_grid(rows =   TrainFunction~TestType,
             labeller = labeller(TrainFunction = tf.labs)) +
  scale_fill_manual(values = color_key$color)+
  geom_boxplot(outlier.shape = 8) +
  theme_bw(base_size = 14) +
  theme(legend.position = "none", 
        axis.text.x= element_text(size = 7, angle = 90, hjust = 1, vjust = 0.5))

ggsave(g2, file = paste0(base_dir, "Best_Models_Precision_byClass.pdf"), 
       width = 19, height = 10, units="in")

g3 =ggplot(best_pr, aes(x=Class, y=Recall, fill = Class)) +
  facet_grid(rows =   TrainFunction~TestType,
             labeller = labeller(TrainFunction = tf.labs)) +
  scale_fill_manual(values = color_key$color)+
  geom_boxplot(outlier.shape = 8) +
  theme_bw(base_size = 14) +
  theme(legend.position = "none", 
        axis.text.x= element_text(size = 7, angle = 90, hjust = 1, vjust = 0.5))

ggsave(g3, file = paste0(base_dir, "Best_Models_Recall_byClass.pdf"), 
       width = 19, height = 10, units="in")
########
for (fun in levels(acc_ind_class$TrainFunction)){
  draw_acc_by_class(acc_ind_class, base_dir = base_dir,
                    col = color_key$color, trainFunction = fun, type = "Inductive")
  draw_acc_by_class(acc_ind_class, base_dir = base_dir,
                    col = color_key$color, trainFunction = fun, type = "Transductive")
}



