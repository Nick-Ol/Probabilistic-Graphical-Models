library("mixtools")

#Importing other .R files in the directory
source("KMeans.R")

dat = as.matrix(read.table("Data/EMGaussian.data", header=F, sep=' '))
init = KMeans(dat, 4)
initCentr = init$centroids



n = length(dat)/2 #TODO fix this

gaussianMixtureEM = function(k) #k is the number of gaussians in the mixture
{
  p = c() #p[i] = P(Y=i)
  mu = c() #means for the k gaussians
  sigma = c() #std dev for the k gaussians
  eta = matrix(nrow=n, ncol=k) #responsabilities for each point (n rows), and every gaussian (k col)
  
  set.seed(1) #seeding for debugging
  randomIndices = sample(1:n, k)
  for (i in 1:k)
  {
    #initializing with arbitrary values
    p[i]=1/k
    mu[i]=dat[randomIndices[i]] 
    sigma[i]=sqrt(var(dat))                    
  }
  iterations = 0
  logLike_new=0
  #what follows is a R version of do while loop
  repeat
  {
    #E step
    for (i in 1:k)
    {
      eta[,i]=p[i]*dnorm(dat, mean=mu[i], sd=sigma[i]) #normalization will come after
    }
    
    #normalizing responsabilities (sum over one line (= one point x) must be 1)
    for (l in 1:n)
    {
      eta[l,] = eta[l,]/sum(eta[l,])
    }
    
    #computing new P(Y=k)
    for (i in 1:k)
    {
      p[i]=sum(eta[,i])/n
    }
    
    #M step
    for (i in 1:k)
    {
      #new means:
      mu[i]=sum(eta[,i]*dat)/sum(eta[,i])
      #new deviations:
      sigma[i]=sqrt(sum(eta[,i]*(dat-mu[i])^2)/sum(eta[,i]))
    }
    logLike_old = logLike_new #the value at the previous step
    logLike_new = 0
    for (i in 1:k)
    {
      #logLike_new = logLike_new + finiteSum(eta[,i]*log(p[i]) + log(dnorm(irm,mu[i],sigma[i])))
      logLike_new = logLike_new + finiteSum(log(p[i]) + log(dnorm(dat,mu[i],sigma[i])))
    }
    
    iterations=iterations+1 #oh wouldn't it be nice to have += or ++
    
    #stopping when relative improvment of LLH is less than 10^-4 %
    #or more than 1000 iterations (something might have gone wrong)    
    #or if one of the variances is to small (overfitting)
    if(iterations>1000 || min(sigma) <1|| abs((logLike_new-logLike_old)/logLike_new) <0.000001)         
    {
      break
    }  
  }
  return (list("p"=p, "mu"=mu, "sigma"=sigma, "iterations"=iterations))
}