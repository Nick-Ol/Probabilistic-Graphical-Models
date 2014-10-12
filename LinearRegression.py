import numpy as np
import matplotlib.pyplot as plt
import math

def LinearRegression(tab):
    X1 = tab[:,0]
    X2 = tab[:,1]
    Y = tab[:,2].astype(int)
    X = np.column_stack((X1,X2))
    Xlr = []
    for i in range(0, len(X)):
        Xlr.append(np.append(X[i],1))
    Xlr = np.asarray(Xlr)
    
    w = np.linalg.lstsq(Xlr,Y)[0]
    
    colors = np.array(['r', 'b'])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X1, X2, c=colors[Y], lw=0) #points cloud. Red = 0, Blue = 1
    print("Linear regression: coef directeur :", -w[0]/w[1], ", ordonnée à l'origine :", -(w[2]-0.5)/w[1])
    x = np.arange(-8,8,0.5)
    ax.plot(x, -w[0]*x/w[1]-(w[2]-0.5)/w[1], c='0') #separator
    plt.xlim([np.min(X1), np.max(X1)])
    plt.ylim([np.min(X2), np.max(X2)])
    plt.title("Linear regression")
    
    return w[0], w[1], w[2]-0.5