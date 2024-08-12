# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 18:33:03 2022

@author: USER
"""

import pandas as pd
import numpy as np
from numpy.linalg import eig
import matplotlib .pyplot as plt
import math

#importing the data
df = pd.read_csv('Dataset.csv', names=['x','y'])

#organising the data
data = np.array([df['x'],df['y']])

#centering in lower dimension
mean_x = np.sum(data[0])/1000
mean_y = np.sum(data[1])/1000

data[0] = np.subtract(data[0], mean_x)
data[1] = np.subtract(data[1], mean_y)

def findKernelPolynomial(d):
    polynomial_K = np.matmul(data.transpose(), data)

    for i in range(0,1000):
        for j in range(0,1000):
            polynomial_K[i][j] = (polynomial_K[i][j] + 1)**d
    return polynomial_K
    
def centerKernel(Kernel):
    one_n = np.zeros([1000,1000])
    for i in range(0,1000):
        for j in range(0,1000):
            one_n[i][j] = (1/1000); 

    one_nK = np.matmul(one_n, Kernel)   
    identity = np.identity(1000)

    i_one_n = np.subtract(identity, one_n)
    Kcentered = np.matmul(i_one_n, Kernel)
    Kcentered = np.matmul(Kcentered, i_one_n)
    return Kcentered

def polynomialPCA(Kcentered):
    eigen_values, eigen_vectors = eig(Kcentered)
    eigen_values = np.real(eigen_values)
    eigen_vectors = eigen_vectors.transpose()
    
    eigen_values_copy = np.real(eigen_values)
    
    max_index = np.argmax(eigen_values)

    max_eigen_value = eigen_values[max_index]
    max_eigen_vector = np.real(eigen_vectors[max_index])

    aplha = np.real(np.divide(max_eigen_vector, math.sqrt(max_eigen_value)))

    eigen_values = np.delete(eigen_values, max_index, axis=0)
    eigen_vectors = np.delete(eigen_vectors, max_index, axis = 0)

    second_max_index = np.argmax(eigen_values)

    second_max_eigen_value = eigen_values[second_max_index]
    second_max_eigen_vector = np.real(eigen_vectors[second_max_index])
    
    print("variance along maximum eigen vector")
    print(max_eigen_value/sum(eigen_values_copy)*100)
    
    print("variance along second maximum eigen vector")
    print(second_max_eigen_value/sum(eigen_values_copy)*100)
    
    aplha_second = np.real(np.divide(second_max_eigen_vector, math.sqrt(second_max_eigen_value)))
    line1 = np.matmul(Kcentered,aplha)
    line2 = np.matmul(Kcentered,aplha_second)
    plt.title("projection of each point in the dataset onto the top-2 components")
    plt.xlabel("along the direcction of principal component1")
    plt.ylabel("along the direcction of principal component2")
    plt.scatter(line1, line2)
    plt.show()
    
#compute the kernel map
Kernel = findKernelPolynomial(2)
print("Kernel PCA for Polynomial Kernel Map function with degree 2")
#center in higher dimension
Kcentered = centerKernel(Kernel)
#perform the PCA
polynomialPCA(Kcentered)



#compute the kernel map
Kernel = findKernelPolynomial(3)
print("Kernel PCA for Polynomial Kernel Map function with degree 3")
#center in higher dimension
Kcentered = centerKernel(Kernel)
#perform the PCA
polynomialPCA(Kcentered)



#Defining function for computing Exponential Kernel
def findKernelExponental(sigma):
    exponental_k = np.zeros([1000,1000])
    for i in range(0,1000):
        for j in range(0,1000):
            x = data[:,i] - data[:,j]
            exponental_k[i][j] = np.matmul(x.transpose(),x)/(2*sigma*sigma)
            exponental_k[i][j] = np.exp(-1*exponental_k[i][j])
    return exponental_k


#Defining function for computing top 2 eigen vectors and finding corresponding projections
def exponentialPCA(Kcentered):
    #eigen decompose the centered Kernel Matrix
    eigen_values, eigen_vectors = eig(Kcentered)
    eigen_values = np.real(eigen_values)
    eigen_vectors = np.real(eigen_vectors.transpose())
    eigen_values_copy = eigen_values
    
    max_index = np.argmax(eigen_values)

    max_eigen_value = eigen_values[max_index]
    max_eigen_vector = eigen_vectors[max_index]

    eigen_values = np.delete(eigen_values, max_index, axis=0)
    eigen_vectors = np.delete(eigen_vectors, max_index, axis = 0)

    second_max_index = np.argmax(eigen_values)
    second_max_eigen_value = eigen_values[second_max_index] 
    second_max_eigen_vector = eigen_vectors[second_max_index]

    print("variance along maximum eigen vector")
    print(max_eigen_value/sum(eigen_values_copy)*100)
    print("variance along second maximum eigen vector")
    print(second_max_eigen_value/sum(eigen_values_copy)*100)
    
    alpha = np.divide(max_eigen_vector, math.sqrt(max_eigen_value))
    aplha_second = np.divide(second_max_eigen_vector, math.sqrt(second_max_eigen_value))
    line1 = np.matmul(Kcentered,alpha)

    line2 = np.matmul(Kcentered,aplha_second)
    
    plt.title("projection of each point in the dataset onto the top-2 components")
    plt.xlabel("along the direcction of principal component1")
    plt.ylabel("along the direcction of principal component2")
    plt.scatter(line1, line2)
    plt.show()
    
#partB
step = 1
for i in range(1,11,step):
    print("Kernel PCA for Exponential Kernel Map function with sigma ", (i/10))
    #compute the kernel map
    Kernel = findKernelExponental(i/10)
    #center in higher dimension
    Kcentered = centerKernel(Kernel)
    #perform the PCA
    exponentialPCA(Kcentered)
    
    

