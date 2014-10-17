# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import math

def LDA(tab):
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
    sigma = np.asmatrix((sigma0 + sigma1)/2) #we take this as global sigma, as we suppose sigma1=sigma2
   
    #compute and plot the separator (black line)
    w = np.squeeze(np.asarray(np.dot(sigma.I,mu1-mu0)))
    b = np.squeeze(np.asarray(math.log(p/float((1-p))) - 0.5*np.dot(np.dot(((mu0+mu1).T),sigma.I),mu1-mu0)))
    w0 = np.squeeze(np.asarray(np.dot(sigma0.I,mu1-mu0)))
    b0 = np.squeeze(np.asarray(math.log(p/float((1-p))) - 0.5*np.dot(np.dot(((mu0+mu1).T),sigma0.I),mu1-mu0)))
    w1 = np.squeeze(np.asarray(np.dot(sigma1.I,mu1-mu0)))
    b1 = np.squeeze(np.asarray(math.log(p/float((1-p))) - 0.5*np.dot(np.dot(((mu0+mu1).T),sigma1.I),mu1-mu0)))

    #display the points cloud with the separator
    colors = np.array(['r', 'b'])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X[:,0], X[:,1], c=colors[Y], lw=0) #points cloud. Red = 0, Blue = 1
    print("LDA : coef directeur :", -w[0]/float(w[1]), ", ordonnee a l'origine :", b/w[1])
    x = np.arange(-8,8,0.5)
    #in black : sigma as (sigma0+sigma1)/2 - green : sigma 0 - yellow : sigma1
    ax.plot(x, -w[0]*x/float(w[1])+b/float(w[1]), c='0')
    ax.plot(x, -w0[0]*x/float(w0[1])+b0/float(w0[1]), c='g')
    ax.plot(x, -w1[0]*x/float(w1[1])+b1/float(w1[1]), c='y')
    plt.xlim([np.min(X[:,0]), np.max(X[:,0])])
    plt.ylim([np.min(X[:,1]), np.max(X[:,1])])
    plt.title("LDA")
    
    return w[0], w[1], -b