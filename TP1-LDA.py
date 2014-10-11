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
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(X1_A, X2_A, c=colors[Y_A], lw=0)

#separate the data according to labels
X10_A = []
X20_A = []
X11_A = []
X21_A = []
for i in range(0,len(Y_A)) :
    if Y_A[i]==0:
        X10_A.append(X1_A[i])
        X20_A.append(X2_A[i])
    else:
        X11_A.append(X1_A[i])
        X21_A.append(X2_A[i])
      
#LDA :
#compute the estimators
p = np.count_nonzero(Y_A)/len(Y_A)
mu0 = np.asarray([np.mean(X10_A), np.mean(X20_A)])
mu1 = np.asarray([np.mean(X11_A), np.mean(X21_A)])
sigma0 = np.cov(X10_A, X20_A)
sigma1 = np.cov(X11_A, X21_A)
sigma = np.asmatrix((sigma0 + sigma1)/2) #we take this as global sigma, as we suppose sigma1=sigma2
#compute and plot the separator (black line)
w = np.dot(sigma.I,mu1-mu0)
b = math.log(p/(1-p)) - 0.5*np.dot(np.dot(((mu0+mu1).T),sigma.I),mu1-mu0)

x = np.arange(-8,8,0.5)
ax1.plot(x, -w[0,0]*x/w[0,1]+b[0,0]/w[0,1], c='0')
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

plt.show()