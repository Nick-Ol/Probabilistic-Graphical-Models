EMllh = function(dat, resultsEM)
{
  n = nrow(dat)
  K = length(resultsEM$p)
  clusters = rep(0,n)
  eta = matrix(nrow=n, ncol=K)
  llh = 0
  
  mu = resultsEM$mu
  variance = resultsEM$sigma
  p = resultsEM$p
  
  for (i in 1:K)
  {
    eta[,i]=p[i]*dmvnorm(dat, mean=mu[[i]], sigma=variance[[i]]) #normalization will come after
  }
  
  for (l in 1:n)
  {
    clusters[l] = which.max(eta[l,]) #attributing points to clusters
  }
  
  for (i in 1:K)
  {
    in_cluster_i = which(clusters == i)
    llh = llh + sum(log(p[i]*dmvnorm(dat[in_cluster_i, ], mean=mu[[i]], sigma=variance[[i]])))
  }
  
  return(llh)
}