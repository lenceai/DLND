# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 06:55:10 2020

@author: datar
"""

#%% 
### 1. Introduction to Deep Learning with PyTorch

import torch


#%%

def activation(x):
    return 1/(1+torch.exp(-x))

#%%
    
torch.manual_seed(7)

features = torch.randn((1, 5))

weights = torch.randn_like(features)

bias = torch.randn((1, 1))

#%%

y = activation(torch.sum(features * weights) + bias)
#y = activation((features * weights).sum() + bias)

#%%

y = activation(torch.mm(features, weights.view(5, 1)) + bias)

#%%

## Generate data
torch.manual_seed(7)

# Features are 3 random variables
features = torch.randn((1, 3))

# Define the size of each layter

n_input = features.shape[1]
n_hidden = 2
n_output = 1

# Weights
W1 = torch.randn(n_input, n_hidden)
W2 = torch.randn(n_hidden, n_output)

# bias

B1 = torch.randn((1, n_hidden))
B2 = torch.randn((1, n_output))

#%%

h = activation(torch.mm(features, W1) + B1)
output = activation(torch.mm(h, W2) + B2)
print(output)


#%%

import numpy as np
a = np.random.rand(4, 3)

b = torch.from_numpy(a)

c = b.numpy()

b.mul_(2)

#%% 
###  2. Neural Networks with PyTorch

#import matplotlib
#config InlineBackend.figure_format = 'retina'

import numpy as np
import torch

import helper

import matplotlib.pyplot as plt

from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),])

trainset = datasets.MNIST('C:/DLND/intro-to-pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

#%%

dataiter = iter(trainloader)
images, labels = dataiter.next()
print(type(images))
print(images.shape)
print(labels.shape)

plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r');

#%%
def activation(x):
    return 1/(1+torch.exp(-x))

inputs = images.view(images.shape[0], -1)

w1 = torch.randn(784, 256)
b1 = torch.randn(256)

w2 = torch.randn(256, 10)
b2 = torch.randn(10)

h = activation(torch.mm(inputs, w1) + b1)

out = torch.mm(h, w2) + b2

#%%

## Solution
def softmax(x):
    return torch.exp(x)/torch.sum(torch.exp(x), dim=1).view(-1, 1)

probabilities = softmax(out)

# Does it have the right shape? Should be (64, 10)
print(probabilities.shape)
# Does it sum to 1?
print(probabilities.sum(dim=1))




