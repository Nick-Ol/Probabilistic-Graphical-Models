# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import math
def QDA(tab):
    #separate data and labels
    X = tab[:,0:2]
    Y = tab[:,2].astype(int)
    
    #separate the data according to labels
    X0 = np.asarray([X[i,:].tolist() for i in range(0,len(X)) if Y[i]==0]) #list comprehension - labels 0
    X1 = np.asarray([X[i,:].tolist() for i in range(0,len(X)) if Y[i]==1]) #labels 1
    
    #compute the estimators
    p = np.count_nonzero(Y)/float(len(Y))
    mu0 = np.asarray([np.mean(X0[:,0]), np.mean(X0[:,1])])
    mu1 = np.asarray([np.mean(X1[:,0]), np.mean(X1[:,1])])
    sigma0 = np.asmatrix(np.cov(X0[:,0], X0[:,1]))
    sigma1 = np.asmatrix(np.cov(X1[:,0], X1[:,1]))
   
    #compute and plot the separator (black line)
    b = np.squeeze(np.asarray(math.log(p/(1-p)) - 0.5*math.log(np.linalg.det(sigma1)/np.linalg.det(sigma0)) - 0.5*np.dot(np.dot(mu1.T,sigma1.I),mu1)
    + 0.5*np.dot(np.dot(mu0.T,sigma0.I),mu0)))
    w = np.squeeze(np.asarray(np.dot(sigma1.I,mu1) - np.dot(sigma0.I,mu0)))
    W = np.asmatrix(- 0.5*(sigma1.I - sigma0.I))

    #display the points cloud with the separator
    colors = np.array(['r', 'b'])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X[:,0], X[:,1], c=colors[Y], lw=0) #points cloud. Red = 0, Blue = 1
    x = np.arange(-8,8,0.02)
    Z1 = np.zeros(len(x))
    Z2 = np.zeros(len(x))
    k=0
    #we are looking for the roots of the quadratic equation
    for xx in x:
        Z1[k] = np.max(np.roots([W[1,1], (W[0,1] + W[1,0])*xx + w[1], W[0,0]*xx**2 + w[0]*xx + b]))
        Z2[k] = np.min(np.roots([W[1,1], (W[0,1] + W[1,0])*xx + w[1], W[0,0]*xx**2 + w[0]*xx + b]))
        k += 1
    
    #plot the curves for the two roots    
    ax.plot(x, Z1)
    ax.plot(x, Z2)
    plt.xlim([np.min(X[:,0]), np.max(X[:,0])])
    plt.ylim([np.min(X[:,1]), np.max(X[:,1])])
    plt.title('QDA')
    #can't find out why three windows open...
    
    return W, w, b