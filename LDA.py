import numpy as np
import matplotlib.pyplot as plt
import math

def LDA(tab):
    X1 = tab[:,0]
    X2 = tab[:,1]
    Y = tab[:,2].astype(int)
    
    #separate the data according to labels
    X10 = []
    X20 = []
    X11 = []
    X21 = []
    for i in range(0,len(Y)):
        if Y[i]==0:
            X10.append(X1[i])
            X20.append(X2[i])
        else:
            X11.append(X1[i])
            X21.append(X2[i])
    
    #compute the estimators
    p = np.count_nonzero(Y)/len(Y)
    mu0 = np.asarray([np.mean(X10), np.mean(X20)])
    mu1 = np.asarray([np.mean(X11), np.mean(X21)])
    sigma0 = np.asmatrix(np.cov(X10, X20))
    sigma1 = np.asmatrix(np.cov(X11, X21))
    sigma = np.asmatrix((sigma0 + sigma1)/2) #we take this as global sigma, as we suppose sigma1=sigma2
   
    #compute and plot the separator (black line)
    w = np.squeeze(np.asarray(np.dot(sigma.I,mu1-mu0)))
    b = np.squeeze(np.asarray(math.log(p/(1-p)) - 0.5*np.dot(np.dot(((mu0+mu1).T),sigma.I),mu1-mu0)))
    w0 = np.squeeze(np.asarray(np.dot(sigma0.I,mu1-mu0)))
    b0 = np.squeeze(np.asarray(math.log(p/(1-p)) - 0.5*np.dot(np.dot(((mu0+mu1).T),sigma0.I),mu1-mu0)))
    w1 = np.squeeze(np.asarray(np.dot(sigma1.I,mu1-mu0)))
    b1 = np.squeeze(np.asarray(math.log(p/(1-p)) - 0.5*np.dot(np.dot(((mu0+mu1).T),sigma1.I),mu1-mu0)))

    colors = np.array(['r', 'b'])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X1, X2, c=colors[Y], lw=0) #points cloud. Red = 0, Blue = 1
    print("LDA : coef directeur :", -w[0]/w[1], ", ordonnée à l'origine :", b/w[1])
    x = np.arange(-8,8,0.5)
    #in black : sigma as (sigma0+sigma1)/2 - green : sigma 0 - yellow : sigma1
    ax.plot(x, -w[0]*x/w[1]+b/w[1], c='0')
    ax.plot(x, -w0[0]*x/w0[1]+b0/w0[1], c='g')
    ax.plot(x, -w1[0]*x/w1[1]+b1/w1[1], c='y')
    plt.xlim([np.min(X1), np.max(X1)])
    plt.ylim([np.min(X2), np.max(X2)])
    plt.title("LDA")
    
    return w[0], w[1], -b