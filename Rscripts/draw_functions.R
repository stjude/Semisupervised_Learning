library(ROCR)
#############
###output: class output from the prediction
###scores: prediction score from the classifier
###colors: color key for 91 classes
###ylab: label for the y-axis
###xlab: label for the x-axis
###title: title of the plot
###metrics: Performance measure to use for the evaluation. 
#A complete list of the performance measures that are available for measure and x.measure is given in the 'Details' section
##of the performance in the ROCR package.
###xlim: x-axis limit range
###ylim: y-axis limit range
###return the threshold analysis
#############

draw_cutoff_curves = function(output, scores, colors, ylab, xlab, title, metrics, xlim, ylim, filename, ...){
  response = as.factor(output$truth_label)
  aucs <- rep(NA, length(levels(response))) # store AUCs
  thres <- rep(NA, length(levels(response)))
  class_thres <- NULL
  
  pdf(file = filename)
  plot(x=NA, y=NA, xlim=xlim, ylim=ylim,
       ylab=ylab,
       xlab=xlab, main = title, 
       bty='n')
  for (i in seq_along(levels(response))) {
    cur.class <- levels(response)[i]
    binary.labels <- as.factor(output$truth_label == cur.class)
    score <- scores[, cur.class] # scores for all samples at the particular class
    test.labels <- output$truth_label == cur.class
    rocobj = roc(test.labels, score)
    
    thres = coords(rocobj, x="all",  best.method="youden", input="threshold", ret=c("threshold", metrics))
    
    if(nrow(thres) > 1){
      num_r = nrow(thres)
      rownames(thres) = paste0(cur.class, "_", seq(num_r))
    }else{
      rownames(thres) = cur.class
    }
    class_thres = rbind(class_thres, thres)
    roc.x <- thres$threshold 
    roc.y <- thres[,2]
    lines(roc.y ~ roc.x,  lwd = 2, col = colors$col[i])
    
  }
#lines(x=c(0,1), c(0,1))
  legend(1,1, levels(response), lty=1, bty="n", col = colors$col, cex=0.8)
  dev.off()
  
  return(class_thres)
}

#fn = paste0(base_dir, toupper(m), "_vs_Cutoff_calibratedScores_perMCF.pdf")
#draw_cutoff_curves(toutput, toutput, colors, ylab = toupper(m), 
 #                  xlab= "Cutoff", title = toupper(m), metrics = m, xlim = c(0,1), ylim = c(0,1), 
#                   filename=fn)
