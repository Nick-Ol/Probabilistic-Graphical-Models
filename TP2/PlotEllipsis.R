library(mvtnorm)

source("EM.R")

plotDataEllipsis = function(dat, k, initCentr, levels, diagonalVariance){
  result = gaussianMixtureEM(dat, k, initCentr, diagonalVariance)
  clusters = result$clusters
  
  plot(dat, col = ifelse(clusters == 1, "red",
                         ifelse(clusters == 2, "blue",
                                ifelse(clusters == 3, "green",
                                       ifelse(clusters == 4, "orange", "black")))),
       pch=16,cex=0.5, asp=1)

  n = 20
  x <- seq(-17, 17, 1/n)
  y <- seq(-17, 17, 1/n)
  
  for (i in 1:k)
  {
    center = result$mu[[i]]
    sigma = as.matrix(result$sigma[[i]])
    points(center[1], center[2], lwd = 2, pch=13, cex =2, col = 'black')
    sigma.inv = solve(sigma, matrix(c(1,0,0,1),2,2))
      
    ellipse <- function(s,t) {u<-c(s,t)-center; u %*% sigma.inv %*% u / 2}
    
    numberPoints = length(x)
    z <- mapply(ellipse, as.vector(rep(x,numberPoints)), as.vector(outer(rep(0,numberPoints), y, `+`)))
    contour(x,y,matrix(z,numberPoints,numberPoints), levels=levels, col = terrain.colors(10), add=TRUE)
  }
}