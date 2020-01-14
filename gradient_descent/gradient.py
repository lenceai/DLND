# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 10:50:10 2020

@author: datar
"""

import numpy as np

def sigmoid(x):
    # Calc sigmoind
    return 1 / (1+np.exp(-x))

def sigmoid_prime(x):
    # Derivative of the sigmoid function
    return sigmoid(x) * (1 - sigmoid(x))

learnrate = 0.5
x = np.array([1,2,3,4])
y = np.array(0.5)

# Initial weights

w = np.array([0.5, -0.5, 0.3, 0.1])

# TODD: calc the nodes linear combination of inputs and weights
h = np.dot(x, w)

# TODO: calc output of the nn
nn_output = sigmoid(h)

# TODO: calc error of the nn
error = y - nn_output

# TODO: calc the error term
error_term = error * sigmoid_prime(h)

# TODO: calc change in weightsw
del_w = learnrate * error_term * x

print('Neural Network output:')
print(nn_output)
print('Amount of Error:')
print(error)
print('Change in Weights:')
print(del_w)