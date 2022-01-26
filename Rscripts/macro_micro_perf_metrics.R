calculate.accuracy <- function(predictions, ref.labels) {
  return(length(which(predictions == ref.labels)) / length(ref.labels))
}
calculate.w.accuracy <- function(predictions, ref.labels, weights) {
  lvls <- levels(ref.labels)
  if (length(weights) != length(lvls)) {
    stop("Number of weights should agree with the number of classes.")
  }
  if (sum(weights) != 1) {
    stop("Weights do not sum to 1")
  }
  accs <- lapply(lvls, function(x) {
    idx <- which(ref.labels == x)
    return(calculate.accuracy(predictions[idx], ref.labels[idx]))
  })
  acc <- mean(unlist(accs))
  return(acc)
}
acc <- calculate.accurac.y(df$Prediction, df$Reference)
print(paste0("Accuracy is: ", round(acc, 2)))

weights <- rep(1 / length(levels(df$Reference)), length(levels(df$Reference)))
w.acc <- calculate.w.accuracy(df$Prediction, df$Reference, weights)
print(paste0("Weighted accuracy is: ", round(w.acc, 2)))

library(caret) # for confusionMatrix function
cm <- vector("list", length(levels(df$Reference)))
for (i in seq_along(cm)) {
  positive.class <- levels(df$Reference)[i]
  # in the i-th iteration, use the i-th class as the positive class
  cm[[i]] <- confusionMatrix(df$Prediction, df$Reference, 
                             positive = positive.class)
}

get.conf.stats <- function(cm) {
  out <- vector("list", length(cm))
  for (i in seq_along(cm)) {
    x <- cm[[i]]
    tp <- x$table[x$positive, x$positive] 
    fp <- sum(x$table[x$positive, colnames(x$table) != x$positive])
    fn <- sum(x$table[colnames(x$table) != x$positie, x$positive])
    # TNs are not well-defined for one-vs-all approach
    elem <- c(tp = tp, fp = fp, fn = fn)
    out[[i]] <- elem
  }
  df <- do.call(rbind, out)
  rownames(df) <- unlist(lapply(cm, function(x) x$positive))
  return(as.data.frame(df))
}

get.micro.f1 <- function(cm) {
  cm.summary <- get.conf.stats(cm)
  tp <- sum(cm.summary$tp, na.rm=TRUE)
  fn <- sum(cm.summary$fn, na.rm=TRUE)
  fp <- sum(cm.summary$fp, na.rm=TRUE)
  pr <- tp / (tp + fp)
  re <- tp / (tp + fn)
  f1 <- 2 * ((pr * re) / (pr + re))
  return(f1)
}
micro.f1 <- get.micro.f1(cm)
print(paste0("Micro F1 is: ", round(micro.f1, 2)))

get.macro.f1 <- function(cm) {
  c <- cm[[1]]$byClass # a single matrix is sufficient
  re <- sum(c[, "Recall"], na.rm=TRUE) / nrow(c)
  pr <- sum(c[, "Precision"], na.rm = TRUE) / nrow(c)
  f1 <- 2 * ((re * pr) / (re + pr))
  return(f1)
}
macro.f1 <- get.macro.f1(cm)
print(paste0("Macro F1 is: ", round(macro.f1, 2)))
