source("KMeans.R")
source("EM.R")
source("PlotEllipsis.R")
source("PlotKMeans.R")
source("EMllh.R")

## PARAMETERS
dat = read.table("Data/EMGaussian.data", header=F, sep=' ')
K = 4 #how many clusters do you want?
levels = (1:9) #which level lines do you want to be plotted?
##


init = KMeans(dat, K, 45)
initCentr = init$centroids
plotClusters(dat, init)
resultsEMDiag = gaussianMixtureEM(dat, K, initCentr, T)
resultsEMNotDiag = gaussianMixtureEM(dat, K, initCentr, F)

par(mfrow=c(1,1))
plotDataEllipsis(dat, resultsEMDiag)
plotDataEllipsis(dat, resultsEMNotDiag)

datTest = read.table("Data/EMGaussian.test", header=F, sep=' ')
cat('llh sur le dataset Test avec covariances diagonales = ', EMllh(datTest, resultsEMDiag) ,'\n')
cat('llh sur le dataset Test avec covariances generales = ', EMllh(datTest, resultsEMNotDiag) ,'\n')



