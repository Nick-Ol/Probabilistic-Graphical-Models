source("KMeans.R")
source("EM.R")
source("PlotEllipsis.R")

## PARAMETERS
dat = read.table("Data/EMGaussian.data", header=F, sep=' ')
K = 4 #how many clusters do you want?
levels = (1:9) #which level lines do you want to be plotted?
##


init = KMeans(dat, K)
initCentr = init$centroids
par(mfrow=c(1,1))
plotDataEllipsis(dat, K, initCentr, levels, T)
plotDataEllipsis(dat, K, initCentr, levels, F)




