# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import math

def LinearRegression(tab):
    #separate data and labels
    X = tab[:,0:2]
    Y = tab[:,2].astype(int)
    #add ones in last column of X, to take the constant of the model into account
    Ones = np.empty(len(X))
    Ones.fill(1)
    Xlr = np.column_stack((X,Ones))
    
    #calculate w = (Xt X)^-1 * Xt Y
    w = np.squeeze(np.asarray(np.dot(np.dot(np.asmatrix(np.dot(Xlr.T,Xlr)).I, Xlr.T), Y)))
    
    #display the points cloud with the separator
    colors = np.array(['r', 'b'])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X[:,0], X[:,1], c=colors[Y], lw=0) #points cloud. Red = 0, Blue = 1
    print("Linear regression: coef directeur :", -w[0]/w[1], ", ordonnee a l'origine :", -(w[2]-0.5)/w[1])
    x = np.arange(-8,8,0.5)
    ax.plot(x, -w[0]*x/w[1]-(w[2]-0.5)/w[1], c='0') #separator
    plt.xlim([np.min(X[:,0]), np.max(X[:,0])])
    plt.ylim([np.min(X[:,1]), np.max(X[:,1])])
    plt.title("Linear regression")
    
    return w[0], w[1], w[2]-0.5