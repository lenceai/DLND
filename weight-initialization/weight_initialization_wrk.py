# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 17:31:27 2020

@author: RyanLence
"""

# Project files for Weight Initialization

#%% Import Libraries 
import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler


#%% Load Data

workers = 0
batch = 100
value_size = 0.2

transform = transforms.ToTensor()

train = datasets.FashionMNIST(root='data', train=True, download=True, transform=transform)
test = datasets.FashionMNIST(root='data', train=False, download = True, transform=transform)

num_train = len(train)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(value_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = torch.utils.data.DataLoader(train, batch_size=batch, sampler=train_sampler, num_workers=workers)
valid_loader = torch.utils.data.DataLoader(train, batch_size=batch, sampler=valid_sampler, num_workers=workers)
test_loader = torch.utils.data.DataLoader(test, batch_size=batch, num_workers=workers)

classes =  ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#%% Visualize Some Training Data

import matplotlib.pyplot as plt
    
# obtain one batch of training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy()

# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(images[idx]), cmap='gray')
    ax.set_title(classes[labels[idx]])

#%% Initialize Weights
    
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, hidden_1=256, hidden_2=128, constant_weight=None):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc3 = nn.Linear(hidden_2, 10)
        self.dropout = nn.Dropout(0.2)
        
        if(constant_weight is not None):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.constant_(m.weight, constant_weight)
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        return x
    
#%%
model_0 = Net(constant_weight = 0)
model_1 = Net(constant_weight = 1)

import helpers

model_list = [(model_0, 'All Zeros'),
              (model_1, 'All Ones')]

helpers.compare_init_weights(model_list, 'Zero vs Ones', train_loader, valid_loader)

#%%
helpers.hist_dist('Random Uniform (low=-3, high=3)', np.random.uniform(-3, 3, [1000]))

#%% Uniform baseline

def weights_init_uniform(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.uniform_(0.0, 1.0)
        m.bias.data.fill_(0)

model_uniform = Net()
model_uniform.apply(weights_init_uniform()
                    
helpers.compare_init_weights([(model_uniform, 'Uniform Weights')], 
                             'Uniform Baseline', 
                             train_loader,
                             valid_loader)
    

