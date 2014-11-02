plotKMeans = function(data, KMeansResults)
{
  centroids = KMeansResults$centroids
  clusters = KMeansResults$clusters
  distortions = KMeansResults$distortions
  
  plot(tab, col = ifelse(clusters == 1, "red",
                         ifelse(clusters == 2, "blue",
                                ifelse(clusters == 3, "green",
                                       ifelse(clusters == 4, "orange", "black")))),
       pch=16,cex=0.5)
  points(centroids, pch=13, cex =2)
  
  plot(distortions)
}