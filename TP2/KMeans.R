tab = read.table("Data/EMGaussian.data", header=F, sep='')

norm = function(X)
{
  return (sqrt(X[1]^2+X[2]^2))
}

KMeans = function(tab, K, seed)
{
  n = nrow(tab)
  set.seed(seed)
  initIndices = sample(1:n, K) #take K random centroids
  centroids = tab[initIndices,] #initialize the centroids
  clusters = rep(0, n) #will contain the cluster associated to the i-th X
  distortions = c(0) #vector with distortion after each iteration
  d = rep(0,K)
  iter =1
  
  repeat
  {
    for (i in 1:n)
    {
      for (j in 1:K)
      {
        d[j] = norm(tab[i,]-centroids[j,]) #vector of distances of Xi to the K centroids
      }
      centroidIndex = min(which(d==min(d))) #cluster of Xi
      clusters[i] = centroidIndex
    }
    
    for (k in 1:K)
    {
      in_cluster_k = which(clusters == k) #indices of the x in cluster j
      #reassign the new centroids
      centroids[k,] = colSums(tab[in_cluster_k,])/length(in_cluster_k)
    }
    
    distortions[iter+1] = 0
    for (i in 1:n)
    {
      distortions[iter+1] = distortions[iter+1] + norm(tab[i,]-centroids[clusters[i],])
    }
    
    #break when the centroids don't change anymore
    if (abs(distortions[iter+1]-distortions[iter]) < .1)
    {
      break
    }

  iter = iter + 1
    
  }
  
  return(list(centroids=centroids, clusters =clusters, distortions=distortions))
}