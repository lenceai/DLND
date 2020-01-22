# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 06:55:10 2020

@author: datar
"""
#%%

import torch


#%%

def activation(x):
    return 1/(1+torch.exp(-x))

#%%
    
torch.manual_seed(7)

features = torch.randn((1, 5))

weights = torch.randn_like(features)

bias = torch.randn((1, 1))

