import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST
import torch.nn as nn
import torch.optim as optim

import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

SEED = 0
TRAIN_PATH = '../data/fraudTrain.csv'
TEST_PATH = '../data/fraudTest.csv'
ADJUSTED_PATH = '../data/fraudAdjusted.csv'
TRAIN_BATCHSIZE = 32 # Fix both batch sizes later
TEST_BATCHSIZE = 64

EPOCHS = 7

# Get cpu or gpu device for training.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using {device} device")

torch.manual_seed(SEED)

# Split a portion of the Train and Test files and use the data to make a Validation file (fraudVal.csv)
dataTrain = pd.read_csv(TRAIN_PATH)
##print(dataTrain)

dataTest = pd.read_csv(TEST_PATH)
dataAdjusted = pd.read_csv(ADJUSTED_PATH)
##print(dataTest)

framePartition = pd.DataFrame(dataAdjusted)
fraudDetected = framePartition['is_fraud'] == '1'
mainVector = torch.Tensor(framePartition.values)

print(mainVector)


# y = data.temp
# X = data.drop('temp', axis=1)



# batch_size = 64

# # Create data loaders.
# train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=2, )
# test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=2, )

# for X, y in test_dataloader:
#     print(f"Shape of X [N, C, H, W]: {X.shape}")
#     print(f"Shape of y: {y.shape} {y.dtype}")
#     break


# Define the architecture of your neural network
# Define the architecture of your neural network
class MyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(30, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 1)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

# Create an instance of your neural network class
net = MyNet()

# Define the loss function and the optimizer
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

# Train the neural network on the training set
for epoch in range(EPOCHS):
    running_loss = 0.0
    for i, data in enumerate(tqdm(mainVector)):
        inputs = torch.tensor(data[:-1], dtype=torch.float32)
        labels = torch.tensor(data[-1], dtype=torch.float32).unsqueeze(0)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1} loss: {running_loss}")