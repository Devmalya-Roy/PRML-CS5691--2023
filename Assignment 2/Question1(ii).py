# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 18:48:03 2022

@author: USER
"""

import pandas as pd
import numpy as np
from numpy.linalg import eig
from numpy.linalg import det
from numpy.linalg import norm
from numpy.linalg import pinv
import matplotlib .pyplot as plt
import math


#importing the data
df = pd.read_csv('A2Q1.csv', header=None)
data = df. to_numpy()

#number of clusters
k_clusters = 4

def calc_lambda(pies, sigma, mean, data, k):
    a = det(sigma[k])
    a = math.sqrt(a)
    a = a * ((2*math.pi)**25)
    a = pies[k]/a
    b = (-1/2)
    b = b * (data[i] - mean[k])
    b = np.matmul(b , pinv(sigma[k]))
    b = np.matmul(b, (data[i] - mean[k]).T)
    return b

def calc_mean(lambdas, data, k):
    nom = 0
    denom = sum(lambdas[:,k])
    for i in range(400):
        nom += lambdas[i][k] * data[i]
    return nom/denom

def calc_sigma(lambdas, data, mean, k):
    nom = 0;
    for i in range(400):
        a = lambdas[i][k]
        b = np.outer(data[i] - mean[k], data[i] - mean[k])
        b = a * b
        nom += b
    return (nom/sum(lambdas[:, k]))

def calc_log_likelihood(data, mean, sigma, pies):
    result = 0
    for i in range(400):
        val = 0
        for k in range(k_clusters):
            a = pies[k]
            b = det(sigma[k])
            b = b * math.sqrt(1 * math.pi)
            b = 1/b
            c = (-1/2)
            c = c * (data[i] - mean[k]).T
            c = c * pinv(sigma[k])
            c = c * (data[i] - mean[k])
            val += a*b*math.exp(det(c))
        result += math.log(val)
    return result

#find objective
def find_objective(lambdas, data): 
    cluster_means = np.zeros([k_clusters,50])
    cluster_count = np.zeros(k_clusters)
    for i in range(400):
        index = np.argmax(lambdas[i])
        cluster_means[index] += data[i]
        cluster_count[index] += 1
    
    for k in range(k_clusters):
        if(cluster_count[k] == 0):
            cluster_count[k] += 1
        cluster_means[k] /= cluster_count[k] 
    
    
    objective = 0
    
    for i in range(400):
        index = np.argmax(lambdas[i])
        objective += norm(data[i] - cluster_means[index])
        
    return objective



#log likelihood
log_likelihood = np.zeros(10)

#em algorithm starts
for itr in range(100):
    #initialize pies
    pies = np.random.random(k_clusters)
    pies = pies/sum(pies)
    
    #initialize random means
    mean = df.sample(k_clusters).to_numpy()
     
    #initialize sigmas
    sigma = []
    data_centred = np.array(data, float)
    for i in range(50):
        data_centred[:,i] -= sum(data_centred[:,i])/400
        
    for k in range(k_clusters):
        sigma.append(np.matmul(data_centred.T, data_centred))
    
    sigma = np.array(sigma)

    for t in range(10):
        #obtain the lambdas
        lambdas = np.zeros([400, k_clusters])
        for i in range(400):
                for k in range(k_clusters):
                    lambdas[i][k] = calc_lambda(pies, sigma, mean, data, k)
                lambdas[i] = lambdas[i]/sum(lambdas[i])
         
        #obtain log likelihood
        log_likelihood[t] += calc_log_likelihood(data, mean, sigma, pies)
        
        #obtain pies for next iteration
        pies_next = np.zeros(k_clusters)
        for k in range(k_clusters):
            pies_next[k] = sum(lambdas[:,k])/400
                
        #obtain means for next iteration
        mean_next = np.zeros([k_clusters,50])
        
        for k in range(k_clusters):
            mean_next[k] = calc_mean(lambdas, data, k)
        
        
        sigma_next = []
        for k in range(k_clusters):
            sigma_next.append(calc_sigma(lambdas, data, mean, k))
        
        sigma = np.array(sigma_next)
        mean = mean_next
        pies = pies_next
    print(itr)

log_likelihood /= 100  
plt.ylabel("Likelihood averaged over 100 randim initializations")
plt.xlabel("iterations") 
plt.plot(log_likelihood)
plt.show()

#objective is 
print(find_objective(lambdas, data))