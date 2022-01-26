checkTrainingData <- function(e){
  e$y <- as.factor(e$y)
  e$x.inst <- as.logical(e$x.inst)
  if(!is.logical(e$x.inst)){
    stop("Parameter x.inst is not logical.")
  }
  if(e$x.inst){
    # Check x
    if(!is.matrix(e$x) && !is.data.frame(e$x)){
      stop("Parameter x is neither a matrix or a data frame.")
    }
    # Check relation between x and y
    if(nrow(e$x) != length(e$y)){
      stop("The rows number of x must be equal to the length of y.")
    }
  }else{
    # Check x
    e$x <- as.matrix(e$x)
    if(!is.matrix(e$x)){
      stop("Parameter x is not a matrix.")
    }
    if(nrow(e$x) != ncol(e$x)){
      stop("The distance matrix x is not a square matrix.")
    } else if(nrow(e$x) != length(e$y)){
      stop(sprintf(paste("The dimensions of the matrix x is %i x %i", 
                         "and it's expected %i x %i according to the size of y."), 
                   nrow(e$x), ncol(e$x), length(e$y), length(e$y)))
    }
  }
}

as.matrix2 <- function(x){
  if(is.matrix(x)){
    return(x)
  }else{
    return(matrix(x, nrow = 1))
  }
}

as.list2 <- function(x, len = 0){
  if(is.null(x)){
    return(vector("list", len))
  }else{
    return(x)
  }
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


#' @title Predictions of the coBC method
#' @description Predicts the label of instances according to the \code{coBC} model.
#' @details For additional help see \code{\link{coBC}} examples.
#' @param object coBC model built with the \code{\link{coBC}} function.
#' @param x An object that can be coerced to a matrix.
#' Depending on how the model was built, \code{x} is interpreted as a matrix 
#' with the distances between the unseen instances and the selected training instances, 
#' or a matrix of instances.
#' @param ... This parameter is included for compatibility reasons.
#' @return Vector with the labels assigned.
#' @export
#' @importFrom stats predict
predict.coBC <- function(object, x, type=c("prob", "class"), ...){
  x <- as.matrix2(x)
  
  ninstances = nrow(x)
  # Predict probabilities per instances using each model
  if(object$x.inst){
    h.prob <- mapply(
      FUN = function(model){
        checkProb(
          predProb(model, x, object$pred, object$pred.pars), 
          ninstances, 
          object$classes
        )
      },
      object$model,
      SIMPLIFY = FALSE
    )
  }else{
    h.prob <- mapply(
      FUN = function(model, indexes){
        checkProb(
          predProb(model, x[, indexes], object$pred, object$pred.pars), 
          ninstances, 
          object$classes
        )
      },
      object$model,
      object$model.index.map,
      SIMPLIFY = FALSE
    )
  }
  if (type == "class"){
    pred <- getClass(
      # Combine probability matrices
      .coBCCombine(h.prob, ninstances, object$classes)
    )
    return(pred)
  }else{
    return(h.prob)
  } 
  
}

#' @title Combining the hypothesis
#' @description This function combines the probabilities predicted by the committee of 
#' classifiers.
#' @param h.prob A list of probability matrices.
#' @param classes The classes in the same order that appear 
#' in the columns of each matrix in \code{h.prob}.
#' @return A probability matrix
#' @export
coBCCombine <- function(h.prob, classes){
  # Check the number of instances
  ninstances <- unique(vapply(X = h.prob, FUN = nrow, FUN.VALUE = numeric(1)))
  if(length(ninstances) != 1){
    stop("The row number of matrixes in the 'pred' parameter are not all equals.") 
  }
  # Check prob matrixes
  vapply(X = h.prob, FUN.VALUE = numeric(),
         FUN = function(prob){
           checkProb(prob, ninstances, classes)
           numeric()
         }
  )
  
  pred <- getClass(
    # Combine probability matrices
    .coBCCombine(h.prob, ninstances, classes)
  )
  
  return(pred)
}

.coBCCombine <- function(h.prob, ninstances, classes){
  
  nclasses <- length(classes)
  
  H.pro <- matrix(nrow = ninstances, ncol = nclasses)
  for(u in 1:ninstances){
    den <- sum(vapply(X = h.prob, FUN = function(prob) sum(prob[u, ]), FUN.VALUE = numeric(1)))
    
    num <- vapply(
      X = 1:nclasses, 
      FUN = function(c){
        sum(vapply(X = h.prob, FUN = function(prob) prob[u, c], FUN.VALUE = numeric(1)))
      }, 
      FUN.VALUE = numeric(1)
    )
    
    H.pro[u, ] <- num / den
  }
  
  colnames(H.pro) <- classes
  
  return(H.pro)
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
predProb <- function(model, x, pred, pred.pars) {
  # Predict probabilities
  ppars <- c(list(model, x), pred.pars)
  # TODO: Call pred function using a try cast function
  prob <- do.call(pred, ppars)
  
  return(prob)
}
