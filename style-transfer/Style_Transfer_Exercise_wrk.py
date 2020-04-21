from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.optim as optim
import requests
from torchvision import transforms, models

#Load in VGG19 

vgg = models.vgg19(pretrained=True).features
for param in vgg.parameters():
    param.requires_grad_(False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

vgg.to(device)

print(vgg)