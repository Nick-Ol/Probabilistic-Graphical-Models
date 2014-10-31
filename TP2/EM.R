library("mixtools")
library("mvtnorm")
#install.packages("mvtnorm")

#Importing other .R files in the directory
source("KMeans.R")

dat = read.table("Data/EMGaussian.data", header=F, sep=' ')
#init = KMeans(dat, 4)
initCentr = matrix(nrow = 4, ncol = 2, c(5, 0, 7, - 7, -2.85 , - 4.25, 5, 3.97))

gaussianMixtureEM = function(dat, k, initCentr) #k is the number of gaussians in the mixture
{
  n = nrow(dat)
  clusters = rep(0,500)
  p = c() #p[i] = P(Y=i)
  mu = list() #means for the k gaussians
  var = list() #std dev for the k gaussians
  eta = matrix(nrow=n, ncol=k) #responsabilities for each point (n rows), and every gaussian (k col)
  
  set.seed(1) #seeding for debugging
  for (i in 1:k)
  {
    #initializing with arbitrary values
    p[i]=1/k
    mu[[i]]=as.matrix(initCentr[i, ])
    var[[i]]=var(sqrt(dat$V1^2 + dat$V2^2)) * diag(2)                    
  }
  iterations = 0
  logLike_new=0
  #what follows is a R version of do while loop
  repeat
  {
    #E step
    for (i in 1:k)
    {
      eta[,i]=p[i]*dmvnorm(dat, mean=mu[[i]], sigma=var[[i]]) #normalization will come after
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
      #new covariances for any covariances:
      var[[i]]=var(dat[in_cluster_i, ]) 
    }
    logLike_old = logLike_new #the value at the previous step
    logLike_new = 0
    for (i in 1:k)
    {
      logLike_new = logLike_new + sum(log(p[i]*dmvnorm(dat, mean=mu[[i]], sigma=var[[i]])))
    }
    
    iterations=iterations+1

    #stopping when relative improvment of LLH is less than 10^-4 %
    #or more than 1000 iterations (something might have gone wrong)
    if(iterations>1000 || abs((logLike_new-logLike_old)/logLike_new) <0.000001)         
    {
      break
    }  
  }
  return (list("p"=p, "mu"=mu, "sigma"=var, "iterations"=iterations))
}

result = gaussianMixtureEM(dat, 4, initCentr)

plot(dat)
for (center in result$mu)
{
  points(center[1], center[2], col = 'blue', lwd = 10)
}

