library(dplyr)

#@labels: list of the class for each sample 
#return: the number of smallest and largest molecular class and the
#number of times should be repeated max/min
get_min_max_class = function(labels){
  out = NULL
  temp = labels %>% group_by(title.class.x)  %>% count(title.class.x)
  min_class = min(temp$n)
  max_class = max(temp$n)
  num_repeat = round(max_class/min_class, 1)
  out = list(min_class, max_class, num_repeat)
  names(out) = c("min_class", "max_class", "num_reapeat")
  return(out)
}

##
#@data: a data frame of beta values: rows are samples, columns are Cpgs
#@labels: list of the class for each sample 
#@seed: seed for random sampling
#@fraction: the proportion of data to keep
#return the index of of the rows (samples) that were randomly selected by the defined fraction
## 

get_fraction_idx = function(data, labels, seed, fraction){
  data = as.data.frame(data)
  data$class = labels
  data$Sample = rownames(data)
  #labels = as.data.frame(labels)
  #colnames(labels) = "class"
  ##randomly sample 80% of the samples in each class
  train_sample = data %>% group_by(class) %>% sample_frac(fraction, seed=seed)
  ##
  idx = match(train_sample$Sample, data$Sample)
  idx = idx[!(is.na(idx))]
  return(idx)
}

###get indices from the training data set after sampling n rows
##@data: a data frame of beta values: rows are samples, columns are Cpgs
###@labels: table of samples with its molecular group
###@replace: TRUE or FALSE to indicate whether sample with or without replacement
###return the idx from the training set

get_sample_n_rows_idx = function(data, labels, n, replace= c("TRUE", "FALSE"), seed){
  data = as.data.frame(data)
  data$class = labels
  data$Sample = rownames(data)
  ###randomly sample n rows from each class in the data set
  temp = data %>% group_by(class) %>% sample_n(size=n, replace=replace, seed = seed)
  #print(head(temp, 20))
  idx = match(data$Sample, temp$Sample)
  idx = idx[!(is.na(idx))]
  return(idx)
}

createConfusionMatrix = function(predict, actual, mode, positive =NULL,...){
  u <- union(predict, actual)
  t <- table(factor(predict, u), factor(actual, u))
  return(caret::confusionMatrix(t, mode= mode, positive = positive))
}

createLabelData = function(sample, predict_label){
  d = data.frame(Sample = sample, class = predict_label)
  return (d)
}

####creat a confusion table matrix from inductive testing set for plotting 
###@m : training model
###@test_x: hold-out test data (inductive test set)
###@test_y: hold-out label data (inductive test set)
###@train_x: training data
###@pred: is the prediction results from a trained model
###@test_y: the hold-out test set
get_inductive_pred = function(m, test_x, test_y, train_x){
  trainer = attr(m, 'class')
  learner = attr(m$model, 'class')
  if (is.null(learner)){
    print(paste("Inductive testing for", trainer))
    #ditest <- as.matrix(proxy::dist(x = test_x, y = train_x, method = "euclidean", by_rows = TRUE))
    #pred <- predict.OneNN(m, ditest[, m$instances.index], type="class")
    pred <- predict(m, test_x)
  }
  else{
    print(paste("Inductive testing for", trainer, "using", learner))
    if (learner == "OneNN"){
      if (trainer == "selfTraining" | trainer == "setred"){
      #ditest <- as.matrix(proxy::dist(x = test_x, y = train_x, method = "euclidean", by_rows = TRUE))
      #ditest is calculated as a global variable in the main program
       #pred <- predict.OneNN(m, ditest[, m$instances.index], type="class")
       pred <- predict(m, ditest[, m$instances.index])
      }
      else{
        print("SNNRCE")
        pred <- predict(m, test_x)
      }
    }else{
      if(trainer == "setred"){
        print("SETRED")
        pred <- predict(m, test_x, type='class')
      }
      else{
        print("ELSE")
        pred <- predict(m, test_x)
      }
    }
  }
  return(pred)
}

####creat a confusion table matrix from transductive testing set for plotting 
###@m : training model
###@train_x: training data set
###@trans_y: reference labels of the unlabeled training data
###@t.na.idx: index of unlabeled samples within the training set as transductive testing
###@pred: is the prediction results from a trained model
###@test_y: the hold-out test set
get_trans_pred = function(m, train_x, trans_y, t.na.idx){
  trainer = attr(m, 'class')
  learner = attr(m$model, 'class')
  
  if (is.null(learner)){
    print(paste("Transductive testing for", trainer))
    #dtrain <- as.matrix(proxy::dist(x = train_x, method = "euclidean", by_rows = TRUE)) 
    #pred <- predict.OneNN(m, dtrain[, m$instances.index], type="class")
    pred <- predict(m, train_x[t.na.idx,])
  }
  else{
    print(paste("Transductive testing for", trainer, "using", learner))
    if (learner == "OneNN"){
      if (trainer == "selfTraining" | trainer == "setred"){
        print("dtrain")
        #dtrain <- as.matrix(proxy::dist(x = train_x, method = "euclidean", by_rows = TRUE)) 
        #dtrain is calculated as a global variable in the main program
        #pred <- predict.OneNN(m, dtrain[t.na.idx, m$instances.index], type="class")
        pred <- predict(m, dtrain[t.na.idx, m$instances.index])
      }
      else{
        print("SNNRCE")
        pred <- predict(m, train_x[t.na.idx,])
      }
    }else{
      if(trainer == "setred"){
        print("SETRED")
        pred <- predict(m, train_x[t.na.idx,], type='class')
      }
      else{
        print("ELSE")
        pred <- predict(m, train_x[t.na.idx,])
      }
    }
  }
  return(pred)
}


####create a table from the cofusionmatrix for plotting
####@cm is the confusion matrix data created by get_CM_table
create_plotTable = function(cm){
  print("Create table for plotting")
  t <- as.data.frame(cm$table)
  names(t) = c("Reference", "Prediction", "Freq")
  plotTable <- t %>%
    mutate(goodbad = ifelse(t$Prediction == t$Reference, "good", "bad")) %>%
    group_by(Reference) %>%
    mutate(prop = Freq/sum(Freq))
  return(plotTable)
}

#Plot the confusion matrix from a model 
###@m: the model used to predict the test set
###@pTable: the CM results from a trained model created by create_CM_table
###base_dir: directory to save the figure
library(ggplot2)
plot_CM = function(pTable, filename){
  print("Plot CM")
  # fill alpha relative to sensitivity/specificity by proportional outcomes 
  #within reference groups (see dplyr code above as well as original confusion matrix for comparison)
  fig = ggplot(data = pTable, mapping = aes(x = Reference, y = Prediction, fill = goodbad, alpha = prop)) +
      geom_tile() +
      geom_text(aes(label = Freq), vjust = .5, alpha = 2, size=2, nudge_x = 0.05, check_overlap = TRUE) +
      scale_fill_manual(values = c(good = "green", bad = "red")) +
      theme_bw()+
      theme(axis.text.x = element_text(angle = -90, hjust=0, vjust=0.5), text=element_text(size=6.5)) +
      xlim(levels(pTable$Reference))
  ggsave(plot=fig, filename = filename)
}
