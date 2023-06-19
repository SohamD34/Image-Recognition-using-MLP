import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import os
import PIL.Image as Image
import pandas as pd
import sklearn
import tensorflow as tf
import keras
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import random_split

# Defining the transformations/augmentations to apply to the data

train_transforms = transforms.Compose([
    transforms.RandomRotation(degrees=5),
    transforms.RandomCrop(size=28, padding=2),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# apply the transformations to the training dataset

mnist_data = datasets.MNIST(root='data', train=True, download=False, transform = train_transforms)
test_data = datasets.MNIST(root='data', train=False, download=False, transform = test_transforms)

# split the dataset into training, testing, and validation sets

train_percent, val_percent = 0.8, 0.2
train_size = int(train_percent * len(mnist_data))
val_size = len(mnist_data) - train_size

train_data, val_data = random_split(mnist_data, [train_size, val_size])


# create the data loaders for each dataset - loader will do all the preprocessing of datasets, loads the data from packages to 
# workable formats and carries out batching.
# Batching - organize the loaded data into batches, enabling efficient processing during training or evaluation. 
# It involves grouping multiple data samples together, which is beneficial for parallel processing and memory optimization.

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)