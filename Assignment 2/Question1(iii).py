# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 22:05:57 2022

@author: USER
"""

import pandas as pd
import numpy as np
from numpy.linalg import eig
from numpy.linalg import det
from numpy.linalg import norm
import matplotlib .pyplot as plt
import math

df = pd.read_csv('A2Q1.csv', header=None)
data = df. to_numpy()

k_cluster = 4

#initialize assignments randomly

def get_random_init(clusters):
    #creating clusters
    cluster_assignment = np.empty([400])

    #Random initialization:
    temp_set = set()
    while(len(temp_set) != clusters):
        cluster_assignment = np.random.randint(0, clusters, 400)
        temp_set = set(cluster_assignment)
        
    return cluster_assignment

z = get_random_init(k_cluster)

final_error = -1
errors = []
mean = np.zeros([k_cluster, 50])
mean_count = np.zeros(k_cluster)

while True:
    for i in range(400):
        cluster = z[i]
        mean[int(cluster)] += data[i]
        mean_count[int(cluster)] += 1
    
    for i in range(k_cluster):
        mean[i] = mean[i]/mean_count[i]
        mean_count[i] = 0
        
        
    error = 0
    for i in range(400):
        cluster = z[i]
        error += norm(data[i] - mean[int(cluster)])
    
    if(final_error == -1):
        final_error = error
    elif(error < final_error):
        #print(final_error - error)
        final_error = error
        errors.append(final_error)
    else:
        break
    
    #reassign
    cluster_new = np.zeros(400)
    for i in range(400):
        
        all_cluster = np.zeros(k_cluster)
        
        for k in range(k_cluster):
            all_cluster[k] = norm(data[i] - mean[k])
            
        cluster_new[i] = np.argmin(all_cluster)
    
    z = cluster_new
 
plt.xlabel("iterations")
plt.ylabel("objective of K-means")
plt.plot(errors)
plt.show()

plt.xlabel("data by index")
plt.ylabel("cluster assignment")
for i in range(400):
    plt.scatter(i, z[i])
    


