# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 00:14:32 2020

@author: datar
"""

import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

# network size
N_input = 4
N_hidden = 3
N_output = 2

np.random.seed(42)

# make some fake data

X = np.random.randn(4)

weights_input_to_hidden = np.random.normal(0, scale=0.1, size=(N_input, N_hidden))
weights_hidden_to_output = np.random.normal(0, scale=0.1, size=(N_hidden, N_output))

# TODO: Make a forward pass throught the netwokr

hidden_layer_in = np.dot(X, weights_input_to_hidden)
hidden_layer_out = sigmoid(hidden_layer_in)

print('Hidden-layer Output:')
print(hidden_layer_out)

output_layer_in = np.dot(hidden_layer_out, weights_hidden_to_output)
output_layer_out = sigmoid(output_layer_in)

print('Output-layer Output:')
print(output_layer_out)