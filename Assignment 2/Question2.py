# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 19:09:45 2022

@author: USER
"""

import pandas as pd
import numpy as np
from numpy.linalg import eig
from numpy.linalg import inv
from numpy.linalg import norm
import matplotlib .pyplot as plt
import math
import random

'''i. Obtain the least squares solution wML to the regression problem using the analytical solution.'''

df = pd.read_csv('A2Q2Data_train.csv', header=None)
data = df. to_numpy()


X= data[:,0:100]
y = data[:, 100]

Wml = np.matmul(X.T, X)
Wml = inv(Wml)
Wml = np.matmul(Wml, X.T)
Wml = np.matmul(Wml, y)

predict_Y_train = np.matmul(X, Wml)
plt.title("Performance of Wml on training data")
plt.xlabel("data by index")
plt.ylabel("corresponding y")
plt.plot(y,'orange')
plt.plot(predict_Y_train,'blue')
plt.legend(['y_train','prediction of y_train'])
print("average train error")
print(norm(predict_Y_train-y)/X.shape[0])


test_data = pd.read_csv('A2Q2Data_test.csv', header=None).to_numpy()
test_X = test_data[:, 0:100]
test_Y = test_data[:, 100]

predict_Y_test = np.matmul(test_X, Wml)

plt.title("Performance of Wml on test data")
plt.xlabel("data by index")
plt.ylabel("corresponding y")
plt.plot(test_Y,'orange')
plt.plot(predict_Y_test,'blue')
plt.legend(['y_test','prediction of y_test'])
plt.show()



print("average test error")
normWml = norm(predict_Y_test-test_Y)/test_Y.shape
print(normWml)

'''ii. Code the gradient descent algorithm with suitable step size to solve the least
squares algorithms and plot kwt − wMLk2 as a function of t. What do you
observe?'''
Wt = np.random.random(100)
errors = []

t = 0
l2 = 0

for t in range(1, 1000):
    direcction = 2*((np.matmul(np.matmul(X.T, X), Wt))-np.matmul(X.T, y));
    direcction = direcction / norm(direcction)
    Wt_next = Wt - (2/t)*direcction
    Wt = Wt_next
    l2 = norm(Wt-Wml)
    #print(l2)
    errors.append(l2)
  
plt.title("The plot of ||Wt - Wml||(L2 norm) as a function of T")    
plt.ylabel("L2 norm of ||Wt - Wml||")
plt.xlabel("t - as number of iterations")
plt.plot(errors,'red')
plt.show()

'''
iii. Code the stochastic gradient descent algorithm using batch size of 100 and plot
kwt − wMLk2 as a function of t. What are your observations?
'''
Wstochastic = np.random.random(100)
error_stochastic = []

for t in range(1,1000):
    #choosing 100 random points 
    points = set()
    while(len(points) != 100):
        points.add(random.randint(0,9999))
        
    Xstochastic = np.zeros([100,100],dtype="float64")
    ystochastic = np.zeros(100,dtype="float64")
    index = 0
    
    for i in points:
        Xstochastic[index] = X[i]
        ystochastic[index] = y[i]
        index = index + 1
        
    #updation of initial Wstochastic
    direcction = 2*((np.matmul(np.matmul(Xstochastic.T, Xstochastic), Wstochastic))-np.matmul(Xstochastic.T, ystochastic))
    direcction = direcction / norm(direcction)
    Wstochastic_new = Wstochastic - (2/t)*direcction
    l2 = norm(Wstochastic_new - Wml)
    error_stochastic.append(l2)
    Wstochastic = Wstochastic_new

plt.title("The plot of ||Wstoc - Wml|| as a function of t")
plt.xlabel("iterations")
plt.ylabel("||Wstoc - Wml||")
plt.plot(error_stochastic)
plt.show()

plt.title("comparision of Stochastic Gradient Descent with Gradient Descent")
plt.xlabel("iterations")
plt.ylabel("difference in L2 norm with Wml")
plt.plot(errors,'red')
plt.plot(error_stochastic,'blue')
plt.legend(['gradient descent','stochastic'])
plt.show()


'''
iv. Code the gradient descent algorithm for ridge regression. Cross-validate for various choices of λ and plot the error in the validation set as a function of λ. For
the best λ chosen, obtain wR. Compare the test error (for the test data in the
file A2Q2Data test.csv) of wR with wML. Which is better and why?
'''

'''
k-fold cross validation
'''

k = 1000
error = []
lambdas = []

for bias in np.arange(0, 2, 0.125): #for each lambda
    
    start = 0
    validation_error = 0
    
    for r in range(10): #for 10 fold cross validation
        X_validation_kfold = X[start:start+k]
        y_validation_kfold = y[start:start+k]
        
        X_train_kfold = X.tolist()
        y_train_kfold = y.tolist()
        
        for i in range(start, start+k):
            X_train_kfold.pop(start)
            y_train_kfold.pop(start)
        
        X_train_kfold = np.array(X_train_kfold)
        y_train_kfold = np.array(y_train_kfold)
        
        
        W_ridge = np.random.random(100)
        for t in range(1, 1000):
            direcction = ((np.matmul(np.matmul(X_train_kfold.T, X_train_kfold), W_ridge))
                            - np.matmul(X_train_kfold.T, y_train_kfold) 
                            + (bias * W_ridge))
            direcction = direcction / norm(direcction)
            Wt_next = W_ridge - (2/t)*direcction
            W_ridge = Wt_next
        
        
        y_test_kfold = np.matmul(X_validation_kfold, W_ridge)
        l2 = norm(y_test_kfold - y_validation_kfold)
        validation_error += l2
        start = start + 1000
    
    validation_error /= 10
    error.append(validation_error)
    lambdas.append(bias)
    
plt.title("average error in the validation set for various choices of lambda")
plt.xlabel("over choices of lambda")
plt.ylabel("error in validation set")
plt.plot(lambdas,error)
print(min(error))
print(np.argmin(error))

lambda_min = 0.125 * (error.index(min(error)))

print("obtained minimum error for lambda:")
print(lambda_min)

W_ridge_closed = np.matmul(X.T, X)
W_ridge_closed = np.add(W_ridge_closed, (lambda_min * np.identity(100)))
W_ridge_closed = inv(W_ridge_closed)
W_ridge_closed = np.matmul(np.matmul(W_ridge_closed, X.T), y)


predict_Y_ridge = np.matmul(test_X, W_ridge_closed)
print("average error for Wridge")
normWridge = norm(predict_Y_ridge-test_Y)/500
print(normWridge)
print("difference between test error of Wridge and Wml")
print(normWridge - normWml)