# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 15:32:16 2020

@author: datar
"""

#%%
import torch
import torchvision
from torchvision import transforms, datasets

train = datasets.MNIST('', train=True, download=True, transform = transforms.Compose([transforms.ToTensor()]))

test = datasets.MNIST('', train=False, download=True, transform = transforms.Compose([transforms.ToTensor()]))


#%%
trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

#%%
for data in trainset:
    print(data)
    break

#%%
x, y = data[0][0], data[1][0]

print(y)

import matplotlib.pyplot as plt

plt.imshow(data[0][0].view(28, 28))
plt.show()

#%%
total = 0
counter_dict = {}

