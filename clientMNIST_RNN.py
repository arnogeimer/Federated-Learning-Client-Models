# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 13:22:45 2022

@author: arno.geimer
"""


import warnings
from collections import OrderedDict

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm
from torch.autograd import Variable
from torch import optim

# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(Net, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)        # Passing in the input and hidden state into the model and  obtaining outputs
        out, hidden = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        #Reshaping the outputs such that it can be fit into the fully connected layer
        out = self.fc(out[:, -1, :])
        return out
    
def train(net, trainloader, epochs):
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr = 0.01)
    sequence_length = 28
    input_size = 28
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(tqdm(trainloader)):
            
            images = images.reshape(-1, sequence_length, input_size).to(DEVICE)
            labels = labels.to(DEVICE)
            
            # Forward pass
            outputs = net(images)
            loss = criterion(outputs, labels)            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
def test(net, testloader):
    """Validate the model on the test set."""
    criterion = nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    sequence_length = 28
    input_size = 28
    with torch.no_grad():
        for images, labels in tqdm(testloader):
            images = images.reshape(-1, sequence_length, input_size).to(DEVICE)
            outputs = net(images.to(DEVICE))
            labels = labels.to(DEVICE)
            loss += criterion(outputs, labels).item()
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    return loss / len(testloader.dataset), correct / total

def load_data():
    """Load MNIST (training and test set)."""
    trainset = MNIST("./data", train=True, download=True, transform=ToTensor())
    testset = MNIST("./data", train=False, download=True, transform=ToTensor())
    return DataLoader(trainset, batch_size=32, shuffle=True), DataLoader(testset)


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10

# Load model and data (simple CNN, CIFAR-10)
net = Net(input_size, hidden_size, num_layers, num_classes).to(DEVICE)
trainloader, testloader = load_data()

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(net, trainloader, epochs=1)
        return self.get_parameters(), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, testloader)
        return loss, len(testloader.dataset), {"accuracy": accuracy}


print("Client starting")

# Start Flower client
fl.client.start_numpy_client("127.0.0.1:8080", client=FlowerClient())

print("Client connecting")
