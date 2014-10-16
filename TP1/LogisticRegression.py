import numpy as np
import matplotlib.pyplot as plt
import math

#define the gradient and the hessian of the log-likelihood l
sigmoid = lambda x: 1/(1+np.exp(-x))
def grad_l(w, Y, X):
    grad = 0
    for i in range(0, len(Y)):
        grad = grad + (Y[i]-sigmoid(np.dot(w.T, X[i])))*X[i]
    return grad
    
def hess_l(w, Y, X):
    hess = 0
    for i in range(0, len(Y)):
        hess = hess + sigmoid(np.dot(w.T, X[i]))*(1-sigmoid(np.dot(w.T, X[i])))*np.outer(X[i],X[i].T)
    return np.asmatrix(hess)
    
#implement Newton-Raphson's method
def newton(Y, X, alpha=1, max_iterations=1000, epsilon=0.001):
    w_old = np.array([1,0,0])
    for i in range(0, max_iterations):
        w_new = np.squeeze(np.asarray(w_old - alpha**i * np.dot(hess_l(w_old,Y,X).I, grad_l(w_old,Y,X))))
        w_new = w_new/np.linalg.norm(w_new,2)
        if np.max(abs(w_new-w_old)) < epsilon:
            break
        w_old = w_new
    return w_new
    
def LogisticRegression(tab):
    X1 = tab[:,0]
    X2 = tab[:,1]
    Y = tab[:,2].astype(int)
    X = np.column_stack((X1,X2))
    Xlr = []
    for i in range(0, len(X)):
        Xlr.append(np.append(X[i],1))
    Xlr = np.asarray(Xlr)
    
    w = newton(Y, Xlr)
    
    colors = np.array(['r', 'b'])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X1, X2, c=colors[Y], lw=0) #points cloud. Red = 0, Blue = 1
    print("Logistic regression: coef directeur :", -w[0]/w[1], ", ordonnée à l'origine :", -w[2]/w[1])
    x = np.arange(-8,8,0.5)
    ax.plot(x, -w[0]*x/w[1]-w[2]/w[1], c='0') #separator
    plt.xlim([np.min(X1), np.max(X1)])
    plt.ylim([np.min(X2), np.max(X2)])
    plt.title("Logistic regression")
    
    return -w[0], -w[1], -w[2]