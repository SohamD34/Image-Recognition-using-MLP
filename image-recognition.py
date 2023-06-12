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


print("Training & Validation data \n", mnist_data)
print("\n\nTesting data \n", test_data)

print("For train_loader -")
print("Dataset loaded =",train_loader.dataset)      
print("Batch size =",train_loader.batch_size)    
print("No. of workers =",train_loader.num_workers)   
print("No. of batches =",len(train_loader))     

print("\nFor val_loader -")
print("Dataset loaded =",val_loader.dataset)      
print("Batch size =",val_loader.batch_size)    
print("No. of workers =",val_loader.num_workers)   
print("No. of batches =",len(val_loader)) 

print("\nFor test_loader -")
print("Dataset loaded =",test_loader.dataset)      
print("Batch size =",test_loader.batch_size)    
print("No. of workers =",test_loader.num_workers)   
print("No. of batches =",len(test_loader)) 

acc = []
layers = []

# Creating Neural Network (MLP) using Sequential Model

# Layers = {ReLU, Sigmoid}

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.Sigmoid(),
    nn.Linear(128,10)
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
epochs = []
all_losses = []

for epoch in range(num_epochs):
    
    epochs.append(epoch)
    print(f"Epoch [{epoch+1}/{num_epochs}]",end="")
    print("xxxxxxxxx",end="")
    for images, labels in train_loader:
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
    print("xxxxxxxxx",end="")
    # Print the loss for every epoch
    print(f", Loss: {loss.item():.4f}")
    all_losses.append(loss.item())
    
    
plt.plot(epochs,all_losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

model.eval()
total_correct = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, dim=1)
        total_correct += (predicted == labels).sum().item()

accuracy = total_correct / len(test_data)
acc.append(accuracy)
layers.append("RS")
print(f"Test Accuracy: {accuracy*100:.2f}%")



# LAYERS = {ReLU, Tanh}

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.Tanh(),
    nn.Linear(128,10)
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
epochs = []
all_losses = []

for epoch in range(num_epochs):
    
    epochs.append(epoch)
    print(f"Epoch [{epoch+1}/{num_epochs}]",end="")
    print("xxxxxxxxx",end="")
    for images, labels in train_loader:
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
    print("xxxxxxxxx",end="")
    # Print the loss for every epoch
    print(f", Loss: {loss.item():.4f}")
    all_losses.append(loss.item())
    
    
plt.plot(epochs,all_losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

model.eval()
total_correct = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, dim=1)
        total_correct += (predicted == labels).sum().item()

accuracy = total_correct / len(test_data)
acc.append(accuracy)
layers.append("RT")
print(f"Test Accuracy: {accuracy*100:.2f}%")


#  LAYERS = {Sigmoid, Tanh}

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.Sigmoid(),
    nn.Linear(256, 128),
    nn.Tanh(),
    nn.Linear(128,10)
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
epochs = []
all_losses = []

for epoch in range(num_epochs):
    
    epochs.append(epoch)
    print(f"Epoch [{epoch+1}/{num_epochs}]",end="")
    print("xxxxxxxxx",end="")
    for images, labels in train_loader:
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
    print("xxxxxxxxx",end="")
    # Print the loss for every epoch
    print(f", Loss: {loss.item():.4f}")
    all_losses.append(loss.item())
    
    
plt.plot(epochs,all_losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

model.eval()
total_correct = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, dim=1)
        total_correct += (predicted == labels).sum().item()

accuracy = total_correct / len(test_data)
acc.append(accuracy)
layers.append("ST")
print(f"Test Accuracy: {accuracy*100:.2f}%")

