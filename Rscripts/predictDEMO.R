as.matrix2 <- function(x){
  if(is.matrix(x)){
    return(x)
  }else{
    return(matrix(x, nrow = 1))
  }
}

#' @title Get classes from a matrix of probabilities and 
#' return the classes indexes
#' @param prob a probabilities matrix
#' @return a vector of indexes corresponding to \code{classes}
#' @noRd
getClassIdx <- function(prob){
  # Obtain classes from probabilities
  map <- apply(prob, MARGIN = 1, FUN = which.max)
  
  return(map)
}

#' @title Check a matrix of probabilities
#' @description Check the number of rows and the columns names
#' of a matrix of probabilities. If the columns are
#' unordered, they are ordered according to \code{classes}.
#' @param prob a probabilities matrix
#' @param ninstances expected number of rows in \code{prob}
#' @param classes expected columns names in \code{prob}
#' @return the matrix \code{prob} with it columns in the order given by \code{classes} 
#' @noRd
checkProb <- function(prob, ninstances, classes){
  # Check probabilities matrix
  if(!is.matrix(prob)){
    stop(
      sprintf(
        paste0(
          "Predict function incorrect output.\n",
          "'prob' is an object of class %s.\n",  
          "Expected an object of class matrix."
        ),
        class(prob)
      )
    )
  }
  if(ninstances != nrow(prob)){
    stop(
      sprintf(
        paste0(
          "Predict function incorrect output.\n",
          "The row number of 'prob' is %s.\n",
          "Expected a number equal to %i (value of 'ninstances')."
        ), 
        nrow(prob), 
        ninstances)
    )
  }
  if(length(classes) != ncol(prob)){
    stop(
      sprintf(
        paste0(
          "Predict function incorrect output.\n",
          "The column number of 'prob' is %s.\n",
          "Expected a number equal to %i (length of 'classes')."
        ), 
        ncol(prob), 
        length(classes))
    )
  }
  if(length(classes) != length(intersect(classes, colnames(prob)))){
    stop(
      paste0(
        "Predict function incorrect output.\n",
        "The columns names of 'prob' is a set not equal to 'classes' set."
      )
    )
  } else {
    # order columns by classes
    prob <- prob[, classes]
    if(!is.matrix(prob)){
      # when nrow of prob is 1
      prob <- matrix(prob, nrow = 1)
      colnames(prob) <- classes
    }
  }
  
  return(prob)
}

#' @title Predict probabilities per classes
#' @param model supervised classifier
#' @param x instances to predict
#' @param pred either a function or a string naming the function for
#' predicting the probabilities per classes, using a base classifier in \code{model}.
#' @param pred.pars A list with parameters that are to be passed to the \code{pred}
#' function.
#' @return a matrix of predicted probabilities
#' @noRd
predProb.demo <- function(model, x, pred, pred.pars) {
  # Predict probabilities
  ppars <- c(list(model, x), pred.pars)
  # TODO: Call pred function using a try cast function
  prob <- do.call(pred, ppars)
  
  return(prob)
}


#' @title Predictions of the Democratic method
#' @description Predicts the label of instances according to the \code{democratic} model.
#' @details For additional help see \code{\link{democratic}} examples.
#' @param object Democratic model built with the \code{\link{democratic}} function.
#' @param x A object that can be coerced as matrix.
#' Depending on how was the model built, \code{x} is interpreted as a matrix 
#' with the distances between the unseen instances and the selected training instances, 
#' or a matrix of instances.
#' @param ... This parameter is included for compatibility reasons.
#' @return Vector with the labels assigned.
#' @export
#' @importFrom stats predict
predict.democratic.prob <- function(object, x, ...){
  x <- as.matrix2(x)
  
  # Select classifiers for prediction
  lower.limit <- 0.5
  selected <- object$W > lower.limit # TODO: create a parameter for 0.5 lower limit
  W.selected <- object$W[selected]
  if(length(W.selected) == 0){
    stop(
      sprintf(
        "%s %s %f",
        "Any classifier selected according model's W values.", 
        "The classifiers are selected when it's W value is greater than",
        lower.limit
      )
    )
  }
  
  # Classify the instances using each classifier
  # The result is a matrix of indexes that indicates the classes
  # The matrix have one column per selected classifier 
  # and one row per instance
  ninstances = nrow(x)
  if(object$x.inst){
    pred <- mapply(
      FUN = function(model, pred, pred.pars){
        getClassIdx(
          checkProb(
            predProb(model, x, pred, pred.pars), 
            ninstances, 
            object$classes 
          )
        ) 
      },
      object$model[selected],
      object$preds[selected],
      object$preds.pars[selected]
    )
  }else{
    pred <- mapply(
      FUN = function(model, indexes, pred, pred.pars){
        getClassIdx(
          checkProb(
            predProb(model, x[, indexes], pred, pred.pars), 
            ninstances, 
            object$classes
          )
        ) 
      },
      object$model[selected], 
      object$model.index.map[selected], 
      object$preds[selected], 
      object$preds.pars[selected]
    )
  }
  pred <- as.matrix2(pred)
  
  
  # Combining predictions
  map <- vector(mode = "numeric", length = ninstances)
  
  for (i in 1:nrow(pred)) {#for each example x in U
    pertenece <- wz <- rep(0, length(object$classes))
    
    for (j in 1:ncol(pred)) {#for each classifier
      z = pred[i, j]
      # Allocate this classifier to group Gz
      pertenece[z] <- pertenece[z] + 1
      wz[z] <- wz[z] + W.selected[j]
    }
    
    # Compute group average mean confidence
    countGj <- (pertenece + 0.5) / (pertenece + 1) * (wz / pertenece)
    map[i] <- which.max(countGj)
  }# end for
  
  cls <- factor(object$classes[map], object$classes)
  n = list(cls, pred)
  return(n)
}
