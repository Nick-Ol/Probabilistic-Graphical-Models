library("mixtools")
library("mvtnorm")
#install.packages("mvtnorm")

source("KMeans.R")

gaussianMixtureEM = function(dat, k, initCentr, diagonalVariance) #k is the number of gaussians in the mixture
{
  colors = c("blue", "green", "brown", "red")
  n = nrow(dat)
  clusters = rep(0,n)
  p = c() #p[i] = P(Y=i)
  mu = list() #means for the k gaussians
  variance = list() #std dev for the k gaussians
  eta = matrix(nrow=n, ncol=k) #responsabilities for each point (n rows), and every gaussian (k col)  
  
  set.seed(1) #seeding for debugging
  for (i in 1:k)
  {
    #initializing with arbitrary values
    p[i]=1/k
    mu[[i]]=as.matrix(initCentr[i, ])
    variance[[i]]=var(sqrt(dat$V1^2 + dat$V2^2)) * diag(2)                    
  }
  logLike_new=0
  
  
  
  for (iterations in 1:1002)
  {
    #E step
    for (i in 1:k)
    {
      eta[,i]=p[i]*dmvnorm(dat, mean=mu[[i]], sigma=variance[[i]]) #normalization will come after
    }
    
    #normalizing responsabilities (sum over one line (= one point x) must be 1)
    for (l in 1:n)
    {
      clusters[l] = which.max(eta[l,]) #attributing points to clusters
      eta[l,] = eta[l,]/sum(eta[l,])
    }
    
    #computing new P(Y=k)
    for (i in 1:k)
    {
      in_cluster_i = which(clusters == i)
      p[i]=length(in_cluster_i)/n
    }
    
    #M step
    for (i in 1:k)
    {
      in_cluster_i = which(clusters == i)
      #new means:
      mu[[i]]=colSums(dat[in_cluster_i, ])/length(in_cluster_i) #colSums
      #points(x= mu[[i]][1], y = mu[[i]][2], col = colors[i], cex =1, lwd = 7)
      #text(mu[[i]], labels = iterations)
      
      #new covariances
      if (diagonalVariance) #if we impose that var be diagonal
      {
        variance[[i]]= var(sqrt(dat[in_cluster_i, ]$V1^2 + dat[in_cluster_i, ]$V2^2)) * diag(2) 
      }
      else
      {
        variance[[i]]=var(dat[in_cluster_i, ]) 
      }
    }
    logLike_old = logLike_new #the value at the previous step
    logLike_new = 0
    for (i in 1:k)
    {
      in_cluster_i = which(clusters == i)
      logLike_new = logLike_new + sum(log(p[i]*dmvnorm(dat[in_cluster_i, ], mean=mu[[i]], sigma=variance[[i]])))
    }
    
    print(iterations)
    #stopping when relative improvment of LLH is less than 10^-4 % or more than 1000 iterations (something might have gone wrong)
    if(iterations>1000 || abs((logLike_new-logLike_old)/logLike_new) <0.000001)         
    {
      break
    }  
  }
  
  return (list("p"=p, "mu"=mu, "sigma"=variance, "iterations"=iterations, "loglike"=logLike_new, "clusters"=clusters))
}