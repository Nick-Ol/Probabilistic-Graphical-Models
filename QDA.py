import numpy as np
import matplotlib.pyplot as plt
import math
def QDA(tab):
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
   
    #compute and plot the separator (black line)
    b = np.squeeze(np.asarray(math.log(p/(1-p)) - 0.5*math.log(np.linalg.det(sigma1)/np.linalg.det(sigma0)) - 0.5*np.dot(np.dot(mu1.T,sigma1.I),mu1)
    + 0.5*np.dot(np.dot(mu0.T,sigma0.I),mu0)))
    w = np.squeeze(np.asarray(np.dot(sigma1.I,mu1) - np.dot(sigma0.I,mu0)))
    W = np.asmatrix(- 0.5*(sigma1.I - sigma0.I))

    colors = np.array(['r', 'b'])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X1, X2, c=colors[Y], lw=0) #points cloud. Red = 0, Blue = 1
    x = np.arange(-8,8,0.02)
    Z1 = np.zeros(len(x))
    Z2 = np.zeros(len(x))
    k=0
    for xx in x:
        Z1[k] = np.max(np.roots([W[0,0], (W[0,1] + W[1,0])*xx + w[1], W[0,0]*xx**2 + w[0]*xx + b]))
        Z2[k] = np.min(np.roots([W[0,0], (W[0,1] + W[1,0])*xx + w[1], W[0,0]*xx**2 + w[0]*xx + b]))
        k += 1
        
    ax.plot(x, Z1)
    ax.plot(x, Z2)
    plt.xlim([np.min(X1), np.max(X1)])
    plt.ylim([np.min(X2), np.max(X2)])
    plt.title('QDA')
    
    return W, w, b