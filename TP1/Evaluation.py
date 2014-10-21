# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import math
from LDA import LDA
from LogisticRegression import LogisticRegression
from LinearRegression import LinearRegression
from QDA import QDA
#go through the three files
filesList = ["A", "B", "C"]
for c in filesList:

     #separate data and labels for train data
    trainData = np.genfromtxt("Data/classification"+c+".train")
    X_train = trainData[:,0:2]
    Y_train = trainData[:,2].astype(int)
    #add ones in last column of X_train
    Ones = np.empty(len(X_train))
    Ones.fill(1)
    Xtilde_train = np.column_stack((X_train,Ones))

    #train the classifiers :
    w_lda = LDA(trainData)
    w_logreg = LogisticRegression(trainData)
    w_linreg = LinearRegression(trainData)
    qda = QDA(trainData)
    W_qda = qda[0]
    w_qda = qda[1]
    b_qda = qda[2]

    #errors on train data
    cpt_lda_train = 0
    cpt_logreg_train = 0
    cpt_linreg_train = 0
    cpt_qda_train = 0
    for i in range(0,len(Xtilde_train)):
        if ((np.dot(w_lda, Xtilde_train[i]) > 0) and (Y_train[i] == 0)) or ((np.dot(w_lda, Xtilde_train[i]) < 0) and (Y_train[i] == 1)):
            cpt_lda_train += 1
        if ((np.dot(w_logreg, Xtilde_train[i]) > 0) and (Y_train[i] == 0)) or ((np.dot(w_logreg, Xtilde_train[i]) < 0) and (Y_train[i] == 1)):
            cpt_logreg_train += 1
        if ((np.dot(w_linreg, Xtilde_train[i]) > 0) and (Y_train[i] == 0)) or ((np.dot(w_linreg, Xtilde_train[i]) < 0) and (Y_train[i] == 1)):
            cpt_linreg_train += 1
        if (((np.dot(X_train[i],np.squeeze(np.asarray(np.dot(W_qda, X_train[i])))) + np.dot(w_qda, X_train[i]) + b_qda > 0) and (Y_train[i] == 0))
        or ((np.dot(X_train[i],np.squeeze(np.asarray(np.dot(W_qda, X_train[i])))) + np.dot(w_qda, X_train[i]) + b_qda < 0) and (Y_train[i] == 1))):
            cpt_qda_train += 1
    print("Size of classification"+c+".train :"+str(len(trainData)))
    print("Number of errors for LDA :"+str(cpt_lda_train))
    print("Number of errors for Logistic regression :"+str(cpt_logreg_train))
    print("Number of errors for Linear regression :"+str(cpt_linreg_train))
    print("Number of errors for QDA :"+str(cpt_qda_train))
    
    #load test data
    testData = np.genfromtxt("Data/classification"+c+".test")
    X_test = testData[:,0:2]
    Y_test = testData[:,2].astype(int)

    #add ones in last column of X_test
    Ones = np.empty(len(X_test))
    Ones.fill(1)
    Xtilde_test = np.column_stack((X_test,Ones))

    #errors on test data
    cpt_lda_test = 0
    cpt_logreg_test = 0
    cpt_linreg_test = 0
    cpt_qda_test = 0
    for i in range(0,len(Xtilde_test)):
        if ((np.dot(w_lda, Xtilde_test[i]) > 0) and (Y_test[i] == 0)) or ((np.dot(w_lda, Xtilde_test[i]) < 0) and (Y_test[i] == 1)):
            cpt_lda_test += 1
        if ((np.dot(w_logreg, Xtilde_test[i]) > 0) and (Y_test[i] == 0)) or ((np.dot(w_logreg, Xtilde_test[i]) < 0) and (Y_test[i] == 1)):
            cpt_logreg_test += 1
        if ((np.dot(w_linreg, Xtilde_test[i]) > 0) and (Y_test[i] == 0)) or ((np.dot(w_linreg, Xtilde_test[i]) < 0) and (Y_test[i] == 1)):
            cpt_linreg_test += 1
        if (((np.dot(X_test[i],np.squeeze(np.asarray(np.dot(W_qda, X_test[i].T)))) + np.dot(w_qda, X_test[i].T) + b_qda > 0) and (Y_test[i] == 0))
        or ((np.dot(X_test[i],np.squeeze(np.asarray(np.dot(W_qda, X_test[i].T)))) + np.dot(w_qda, X_test[i].T) + b_qda < 0) and (Y_test[i] == 1))):
            cpt_qda_test += 1
    print("Size of classification"+c+".test :"+str(len(testData)))
    print("Number of errors for LDA :"+str(cpt_lda_test))
    print("Number of errors for Logistic regression :"+str(cpt_logreg_test))
    print("Number of errors for Linear regression :"+str(cpt_linreg_test))
    print("Number of errors for QDA :"+str(cpt_qda_test))

    #display points cloud and separators for train data
    #to do : test data
    plt.show()