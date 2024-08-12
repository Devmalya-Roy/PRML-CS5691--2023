# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 13:19:51 2022

@author: USER
"""

import pandas as pd
import numpy as np
import matplotlib .pyplot as plt
import math
from numpy.linalg import norm


#importing the data

df = pd.read_csv('A2Q1.csv', header=None)
data = df. to_numpy()


k_cluster = 4

def calculate_lambda(prob, pies, data, k):
    a = pies[k]
    for d in range(50):
        a = a * ((prob[k][d]**data[d]) * ((1-prob[k][d])**(1 - data[d])))
    
    return a

def calculate_probabilities(lambdas, data, k, d):
    res = 0
    
    for i in range(400):
        res += lambdas[i][k] * data[i][d]
        
    return res/sum(lambdas[:, k])

def calculate_log_likelihood(pies, prob, data):
    val = 0
    for i in range(400):
        total = 0
        for k in range(k_cluster):
            a = pies[k]
            for d in range(50):
                a = a * (prob[k][d]**data[i][d]) * ((1-prob[k][d])**(1-data[i][d]))
            total += a
        val += math.log(total)
    return val

#find objective
def find_objective(lambdas, data): 
    cluster_means = np.zeros([k_cluster,50])
    cluster_count = np.zeros(k_cluster)
    for i in range(400):
        index = np.argmax(lambdas[i])
        cluster_means[index] += data[i]
        cluster_count[index] += 1
    
    for k in range(k_cluster):
        if(cluster_count[k] == 0):
            cluster_count[k] += 1
        cluster_means[k] /= cluster_count[k] 
    
    
    objective = 0
    
    for i in range(400):
        index = np.argmax(lambdas[i])
        objective += norm(data[i] - cluster_means[index])
        
    return objective



log_likelihood = np.zeros(10)
objectives = []

for itr in range(20):
    
    #initializing probabilities
    prob = np.zeros([k_cluster,50])
    for i in range(k_cluster):
        prob[i] = np.random.random(50)
    
    #initializing pies
    pies = np.random.random(k_cluster)
    pies = pies/sum(pies)
 
    for step in range(10):
        #estimate lambda
        lambdas = np.zeros([400, k_cluster])
        
        for i in range(400):
            for k in range(k_cluster):
                lambdas[i][k] = calculate_lambda(prob, pies, data[i], k)
            lambdas[i] = lambdas[i]/sum(lambdas[i])
            
        log_likelihood[step] += calculate_log_likelihood(pies, prob, data)
        
        #obtain pies for next iteration
        pie_next = np.zeros(k_cluster)
        for k in range(k_cluster):
            pie_next[k] = sum(lambdas[:,k])/400
        
        #obtain probabilities for next iteration
        prob_next = np.zeros([k_cluster, 50])
        for k in range(k_cluster):
            for d in range(50):
                prob_next[k][d] = calculate_probabilities(lambdas, data, k, d)
        
        prob = prob_next
        pies = pie_next
    objectives.append(find_objective(lambdas, data))
    print(itr)

log_likelihood = log_likelihood/100
plt.title("Log likelihood as a function of iterations")
plt.ylabel("Likelihood averaged over 100 randim initializations")
plt.xlabel("iterations") 
plt.plot(log_likelihood)
plt.show()

#average objective over 100 random iterations
print(sum(objectives)/100)
