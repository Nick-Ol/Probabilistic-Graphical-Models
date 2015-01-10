# -*- coding: utf-8 -*-
"""
Created on Thu Dec 25 22:56:00 2014

@author: Clement Nicolle
"""

from scipy.stats import multivariate_normal
import numpy as np


def gaussian_emission_proba(state, output, means, covariances):
    normal_law = multivariate_normal(means[state], covariances[state])
    return normal_law.pdf(output)


# alpha(t,state) = P(u(1),u(2),...,u(t),q(t)=state)
# alpha(t,state) = alpha_scaled(t,state) / Prod from 0 to t-1 of scale_alpha[i]
def forward(obs_sequence, states,
            start_proba, transition_proba, means, covariances):

    N = len(obs_sequence)
    alpha_scaled = [{}]
    scale_alpha = np.zeros(N)
    # initialization :
    for state in states:
        alpha_scaled[0][state] = start_proba[state] * \
            gaussian_emission_proba(state, obs_sequence[0],
                                    means, covariances)
    scale_alpha[0] = 1 / sum(alpha_scaled[0].values())
    alpha_scaled[0] = {k: alpha_scaled[0][k] * scale_alpha[0]
                       for k in alpha_scaled[0].viewkeys()}
    # recursion :
    for i in range(1, N):
        alpha_scaled.append({})
        for state_to in states:
            prob = 0
            for state_from in states:
                prob += alpha_scaled[i-1][state_from] * \
                    transition_proba[state_from, state_to]
            alpha_scaled[i][state_to] = prob * \
                gaussian_emission_proba(state_to, obs_sequence[i],
                                        means, covariances)

        scale_alpha[i] = 1 / sum(alpha_scaled[i].values())
        alpha_scaled[i] = {k: alpha_scaled[i][k] * scale_alpha[i]
                           for k in alpha_scaled[i].viewkeys()}

    return alpha_scaled, scale_alpha


# beta(t,state) = P(u(t+1),u(t+2),...u(T) | q(t)=state)
# beta(t,state) = beta_scaled(t,state) / Prod from N-1 to t of alpha_scaled[i]
def backward(obs_sequence, states, transition_proba, means, covariances,
             scale_alpha):

    N = len(obs_sequence)
    beta_scaled = [{}]
    # initilization :
    for state in states:
        beta_scaled[0][state] = 1 * scale_alpha[N-1]
    # recursion :
    for i in range(1, N):
        beta_scaled.insert(0, {})
        for state_from in states:
            prob = 0
            for state_to in states:
                prob += transition_proba[state_from, state_to] * \
                    beta_scaled[1][state_to] * \
                    gaussian_emission_proba(state_to, obs_sequence[N-i],
                                            means, covariances)
            beta_scaled[0][state_from] = prob * scale_alpha[N-1-i]

    return beta_scaled


# gamma(t, state) = P(q(t)=state | u(1)...u(T))
def gammas(obs_sequence, states, alpha_scaled, beta_scaled, scale_alpha):

    N = len(obs_sequence)
    gamma = []
    for t in range(N):
        gamma.append({})
        for state in states:
            gamma[t][state] = alpha_scaled[t][state] * \
                beta_scaled[t][state] / scale_alpha[t]

    return gamma


# xi(t, state1, state2) = P(q(t)=state1, q(t+1)=state2 | u(1)...u(T))
def xis(obs_sequence, states,
        transition_proba, means, covariances, alpha_scaled, beta_scaled):

    N = len(obs_sequence)
    xi = []
    for t in range(N-1):
        xi.append({})
        for state_from in states:
            xi[t][state_from] = {}
            for state_to in states:
                xi[t][state_from][state_to] = alpha_scaled[t][state_from] * \
                    transition_proba[state_from, state_to] * \
                    gaussian_emission_proba(state_to, obs_sequence[t+1],
                                            means, covariances) * \
                    beta_scaled[t+1][state_to]

    return xi


def update_params(obs_sequence, states, gamma, xi):

    S = len(states)
    N = len(obs_sequence)
    d = len(obs_sequence[0])  # dimension
    start_proba = np.zeros(S)
    transition_proba = np.zeros((S, S))
    means = np.zeros((S, d))
    covariances = np.zeros((S, d, d))
    for state in states:
        # start_proba
        start_proba[state] = gamma[0][state]

        # transition_proba
        gamma_sum = 0
        for t in range(N-1):
                gamma_sum += gamma[t][state]
        for state_to in states:
            xi_sum = 0
            for t in range(N-1):
                xi_sum += xi[t][state][state_to]
            transition_proba[state][state_to] = xi_sum / gamma_sum

        # means
        gamma_mean = 0
        gamma_sum = 0
        for t in range(N):
            gamma_mean += gamma[t][state] * obs_sequence[t]
            gamma_sum += gamma[t][state]
        means[state] = gamma_mean / gamma_sum

        # covariances
        gamma_covariances = 0
        for t in range(N):
            gamma_covariances += gamma[t][state] * \
                np.outer(obs_sequence[t] - means[state],
                         obs_sequence[t] - means[state])
        covariances[state] = gamma_covariances / gamma_sum

    return start_proba, transition_proba, means, covariances


# logllh = log P(u(1),u(2),...,u(t))
# with alpha_scaled we have : logllh =
# log(sum over states of alpha_scaled[N-1]) - sum over t of (scale_alpha[t])
def loglike(states, alpha_scaled, scale_alpha):

    N = len(alpha_scaled)
    alpha_scaled_sum = 0
    for state in states:
        alpha_scaled_sum += alpha_scaled[-1][state]
    logllh = np.log(alpha_scaled_sum)
    for t in range(N):
        logllh -= np.log(scale_alpha[t])
    return logllh


def baum_welch(obs_sequence, states, start_proba_init,
               transition_proba_init, means_init, covariances_init,
               delta=1e-4):

    iteration = 0
    logllh = []
    start_proba = start_proba_init
    all_start_probas = [start_proba_init]
    transition_proba = transition_proba_init
    all_transition_probas = [transition_proba_init]
    means = means_init
    all_means = [means_init]
    covariances = covariances_init
    all_covariances = [covariances_init]
    while ((iteration == 0) or (iteration == 1)
           or (abs(logllh[iteration-1] - logllh[iteration-2]) > delta)):
        iteration += 1
        print "Iteration %d" % (iteration)
        # forward
        alpha_scaled, scale_alpha = forward(obs_sequence, states,
                                            start_proba, transition_proba,
                                            means, covariances)
        # backward
        beta_scaled = backward(obs_sequence, states, transition_proba,
                               means, covariances, scale_alpha)
        # evaluate gamma and xi
        gamma = gammas(obs_sequence, states, alpha_scaled,
                       beta_scaled, scale_alpha)
        xi = xis(obs_sequence, states, transition_proba,
                 means, covariances, alpha_scaled, beta_scaled)
        # learn the new parameters
        (start_proba, transition_proba,
         means, covariances) = update_params(obs_sequence, states, gamma, xi)
        all_start_probas.append(start_proba)
        all_transition_probas.append(transition_proba)
        all_means.append(means)
        all_covariances.append(covariances)
        # compute the new log-likelihood
        logllh.append(loglike(states, alpha_scaled, scale_alpha))

    logllh = np.asarray(logllh)

    return (all_start_probas, all_transition_probas, all_means,
            all_covariances, logllh, iteration)


def viterbi(obs_sequence, states,
            start_proba, transition_proba,
            means, covariances):

    N = len(obs_sequence)
    V = [{}]
    path = {}
    # Initialize base cases (t == 0)
    for state in states:
        V[0][state] = np.log(start_proba[state] *
                             gaussian_emission_proba(state, obs_sequence[0],
                                                     means, covariances))
        path[state] = [state]

    # Run Viterbi for t > 0
    for t in range(1, N):
        V.append({})
        newpath = {}
        for state in states:
            (prob, best_state_from) = \
                max((V[t-1][state_from] *
                     np.log(transition_proba[state_from][state]) *
                     np.log(gaussian_emission_proba(state, obs_sequence[t],
                                                    means, covariances)),
                    state_from) for state_from in states)
            V[t][state] = prob
            newpath[state] = path[best_state_from] + [state]

        path = newpath  # No need to remember old paths
    (prob, best_last_state) = max((V[N-1][last_state], last_state)
                                  for last_state in states)
    return path[best_last_state]
