# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import math
#load the data
tab = np.genfromtxt("Data/classificationA.train")
X1_A = tab[:,0]
X2_A = tab[:,1]
Y_A = tab[:,2].astype(int)

#display points cloud with labels. Red = 0, Blue = 1
colors = np.array(['r', 'b'])
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.scatter(X1_A, X2_A, c=colors[Y_A], lw=0)

#separate the data according to labels
X10_A = []
X20_A = []
X11_A = []
X21_A = []
for i in range(0,len(Y_A)):
    if Y_A[i]==0:
        X10_A.append(X1_A[i])
        X20_A.append(X2_A[i])
    else:
        X11_A.append(X1_A[i])
        X21_A.append(X2_A[i])
      
#LDA :
#compute the estimators
p = np.count_nonzero(Y_A)/float(len(Y_A))
mu0 = np.asarray([np.mean(X10_A), np.mean(X20_A)])
mu1 = np.asarray([np.mean(X11_A), np.mean(X21_A)])
sigma0 = np.asmatrix(np.cov(X10_A, X20_A))
sigma1 = np.asmatrix(np.cov(X11_A, X21_A))
sigma = np.asmatrix((sigma0 + sigma1)/2) #we take this as global sigma, as we suppose sigma1=sigma2
#compute and plot the separator (black line)
w = np.dot(sigma.I,mu1-mu0)
b = math.log(p/(1-p)) - 0.5*np.dot(np.dot(((mu0+mu1).T),sigma.I),mu1-mu0)
w0 = np.dot(sigma0.I,mu1-mu0)
b0 = math.log(p/(1-p)) - 0.5*np.dot(np.dot(((mu0+mu1).T),sigma0.I),mu1-mu0)
w1 = np.dot(sigma1.I,mu1-mu0)
b1 = math.log(p/(1-p)) - 0.5*np.dot(np.dot(((mu0+mu1).T),sigma1.I),mu1-mu0)

print("LDA : coef directeur :", -w[0,0]/w[0,1], ", ordonnée à l'origine :", b[0,0]/w[0,1])
x = np.arange(-8,8,0.5)
#in black : sigma as (sigma0+sigma1)/2 - green : sigma 0 - yellow : sigma1
ax1.plot(x, -w[0,0]*x/w[0,1]+b[0,0]/w[0,1], c='0')
ax1.plot(x, -w0[0,0]*x/w0[0,1]+b0[0,0]/w0[0,1], c='g')
ax1.plot(x, -w1[0,0]*x/w1[0,1]+b1[0,0]/w1[0,1], c='y')
plt.title("LDA")

#using scikit-learn :
from sklearn.lda import LDA
X_A = np.column_stack((X1_A,X2_A))
clf = LDA()
clf.fit(X_A, Y_A)
xx, yy = np.meshgrid(np.arange(-8, 8, .02),
                         np.arange(-5, 5, .02))
Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
cm = plt.cm.RdBu
Z = Z.reshape(xx.shape)
ax1.contourf(xx, yy, Z, cmap=cm, alpha = 0.8)
#the true separator is quadratic as sigma0 is not exactly equal to sigma1

#Logistic Regression
#adding the third coordinate 1 to X
Xlr_A = []
for i in range(0, len(X_A)):
    Xlr_A.append(np.append(X_A[i],1))
Xlr_A = np.asarray(Xlr_A)
#define the gradient and the hessian of the log-likelihood l
sigmaid = lambda x: 1/(1+np.exp(-x))
def grad_l(w, Y, X):
    grad = 0
    for i in range(0, len(Y)):
        grad = grad + (Y[i]-sigmaid(np.dot(w.T, X[i])))*X[i]
    return grad

def hess_l(w, Y, X):
    hess = 0
    for i in range(0, len(Y)):
        hess = hess + sigmaid(np.dot(w.T, X[i]))*(1-sigmaid(np.dot(w.T, X[i])))*np.outer(X[i],X[i].T)
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

w = newton(Y_A, Xlr_A)
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.scatter(X1_A, X2_A, c=colors[Y_A], lw=0) #points cloud
print("Logistic regression: coef directeur :", -w[0]/w[1], ", ordonnée à l'origine :", w[2]/w[1])
x = np.arange(-8,8,0.5)
ax2.plot(x, -w[0]*x/w[1]-w[2]/w[1], c='0') #separator
plt.title("Logistic regression")

plt.show()