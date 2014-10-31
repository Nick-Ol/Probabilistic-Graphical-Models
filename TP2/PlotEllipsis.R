library(mvtnorm)

source("EM.R")

plotDataEllipsis = function(dat, k, initCentr, levels, diagonalVariance){
  result = gaussianMixtureEM(dat, k, initCentr, diagonalVariance)
  plot(dat, asp= 1)
  n = 20
  x <- seq(-10, 10, 1/n)
  y <- seq(-10, 10, 1/n)
  
  for (i in 1:k)
  {
    center = result$mu[[i]]
    sigma = as.matrix(result$sigma[[i]])
      
    sigma.inv = solve(sigma, matrix(c(1,0,0,1),2,2))
      
    ellipse <- function(s,t) {u<-c(s,t)-center; u %*% sigma.inv %*% u / 2}
    
    numberPoints = length(x)
    z <- mapply(ellipse, as.vector(rep(x,numberPoints)), as.vector(outer(rep(0,numberPoints), y, `+`)))
    contour(x,y,matrix(z,numberPoints,numberPoints), levels=levels, col = terrain.colors(10), add=TRUE)
  }
}