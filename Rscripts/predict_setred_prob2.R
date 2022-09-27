library(ssc)

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
    ssc::checkProb(
      predProb(object$model, x, object$pred, object$pred.pars), 
      ninstances = nrow(x), 
      object$classes
    )
  
  return(result)
}
