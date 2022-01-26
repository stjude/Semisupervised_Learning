#' @title Predict probabilities per classes
#' @param model supervised classifier
#' @param x instances to predict
#' @param pred either a function or a string naming the function for
#' predicting the probabilities per classes, using a base classifier in \code{model}.
#' @param pred.pars A list with parameters that are to be passed to the \code{pred}
#' function.
#' @return a matrix of predicted probabilities
#' @noRd
predProb <- function(model, x, pred, pred.pars) {
  # Predict probabilities
  ppars <- c(list(model, x), pred.pars)
  # TODO: Call pred function using a try cast function
  prob <- do.call(pred, ppars)
  
  return(prob)
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

as.matrix2 <- function(x){
  if(is.matrix(x)){
    return(x)
  }else{
    return(matrix(x, nrow = 1))
  }
}


predict.setred.prob <- function(object, x, ...) {
  x <- as.matrix2(x)
  result <- 
    checkProb(
      predProb(object$model, x, object$pred, object$pred.pars), 
      ninstances = nrow(x), 
      object$classes
    )
  
  return(result)
}
