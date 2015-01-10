# -*- coding: utf-8 -*-
"""
Created on Fri Dec 26 15:24:00 2014

@author: Clement Nicolle
"""

from __future__ import division
import numpy as np
import hmm
import matplotlib.pyplot as plt

data_train = np.loadtxt("../Data/EMGaussian.data")
data_test = np.loadtxt("../Data/EMGaussian.test")

means_init = np.asarray([[-2.03, 4.17], [3.98, 3.77],
                         [3.80, -3.80], [-3.06, -3.53]])
cov0 = np.asarray([[2.90, 0.21], [0.21, 2.76]])
cov1 = np.asarray([[0.21, 0.29], [0.29, 12.24]])
cov2 = np.asarray([[0.92, 0.06], [0.06, 1.87]])
cov3 = np.asarray([[6.24, 6.05], [6.05, 6.18]])
covariances_init = np.asarray([cov0, cov1, cov2, cov3])

means_init_bis = np.asarray([[-2.26, 4.33], [3.97, 6.70],
                             [2.45, -1.45], [-4.48, -4.92]])
cov_bis0 = 2.36 * np.eye(2)
cov_bis1 = 1.65 * np.eye(2)
cov_bis2 = 6.57 * np.eye(2)
cov_bis3 = 3.18 * np.eye(2)
covariances_init_bis = np.asarray([cov_bis0, cov_bis1, cov_bis2, cov_bis3])

states = [0, 1, 2, 3]
start_proba_init = np.ones(4)/4
transition_proba_init = np.asarray([[1/2, 1/6, 1/6, 1/6],
                                    [1/6, 1/2, 1/6, 1/6],
                                    [1/6, 1/6, 1/2, 1/6],
                                    [1/6, 1/6, 1/6, 1/2]])

# 2
alpha_scaled, scale_alpha = hmm.forward(data_test, states,
                                        start_proba_init,
                                        transition_proba_init,
                                        means_init, covariances_init)
beta_scaled = hmm.backward(data_test, states, transition_proba_init,
                           means_init, covariances_init, scale_alpha)

gamma = hmm.gammas(data_test, states, alpha_scaled,
                   beta_scaled, scale_alpha)

for i in states:
    y = np.zeros(100)
    for t in range(100):
        y[t] = gamma[t][i]
    plt.figure()
    plt.plot(y)
    plt.title("State %i" % (i+1))

# 4
(start_proba1, transition_proba1, means1, covariances1,
 logllh1, iteration1) = hmm.baum_welch(data_train, states, start_proba_init,
                                       transition_proba_init,
                                       means_init, covariances_init,
                                       delta=1e-4)
# 5
plt.figure()
plt.plot(logllh1)

logllh2 = []
for i in range(iteration1+1):
    alpha_scaled2, scale_alpha2 = hmm.forward(data_test, states,
                                              start_proba1[i],
                                              transition_proba1[i],
                                              means1[i],
                                              covariances1[i])
    logllh2.append(hmm.loglike(states, alpha_scaled2, scale_alpha2))
plt.figure()
plt.plot(logllh2)

# 6
print "The log-likelihood for HMM on train data is %f" % (logllh1[-1])
print "The log-likelihood for HMM on test data is %f" % (logllh2[-1])

# 7
path1 = hmm.viterbi(data_train, states,
                    start_proba1[-1], transition_proba1[-1],
                    means1[-1], covariances1[-1])


def plotViterbi(data, path, means):
    n = len(data)
    K = len(means)
    colors = ['b', 'g', 'r', 'y']
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i in range(0, n):
        cluster = int(path[i])
        ax.scatter(data[i, 0], data[i, 1], color=colors[cluster])
    for j in range(0, K):
        ax.scatter(means[j, 0], means[j, 1], color="black")

plt.figure()
plotViterbi(data_train, path1, means1[-1])
plt.title("Train data")

# 8
alpha_scaled, scale_alpha = hmm.forward(data_test, states,
                                        start_proba1[-1],
                                        transition_proba1[-1],
                                        means1[-1],
                                        covariances1[-1])
beta_scaled = hmm.backward(data_test, states, transition_proba1[-1],
                           means1[-1], covariances1[-1], scale_alpha)
gamma = hmm.gammas(data_test, states, alpha_scaled,
                   beta_scaled, scale_alpha)
for i in states:
    y = np.zeros(100)
    for t in range(100):
        y[t] = gamma[t][i]
    plt.figure()
    plt.ylim((0, 1))
    plt.plot(y)
    plt.title("State %i" % (i+1))

# 9
most_likely_state = []
for t in range(100):
    most_likely_state.append(np.argmax(gamma[t].values()))
plt.figure()
plt.ylim((-1, 4))
plt.plot(most_likely_state)

# 10
path2 = hmm.viterbi(data_test, states,
                    start_proba1[-1], transition_proba1[-1],
                    means1[-1], covariances1[-1])
plt.figure()
plotViterbi(data_test, path2, means1[-1])
plt.title("Test data")

plt.figure()
plt.ylim((-1, 4))
plt.plot(path2[:100])
# Comparison :
plt.figure()
plt.ylim((-1, 4))
plt.plot(np.asarray(most_likely_state)-np.asarray(path2[:100]))
