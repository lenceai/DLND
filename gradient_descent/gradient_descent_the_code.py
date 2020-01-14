# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 08:13:26 2020

@author: datar
"""
#%%

import numpy as np

#define sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Derivative of the sigmoid function
    
def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

# imput the data
    
x = np.array([0.1, 0.3])

y = 0.2

weights = np.array([-0.8, 0.5])

# Learning rate

learnrate = 0.5

h = x[0]*weights[0] + x[1]*weights[1]

# The neural network output

nn_output = sigmoid(h)

# output error

error = y - nn_output

output_grad = sigmoid_prime(h)

error_term = error * output_grad

del_w = [ learnrate * error_term * x[0],
          learnrate * error_term * x[1]]

#%%


