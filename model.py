import torch
import torch.nn as nn
import torch.nn.functional as F  # Import functional module

class SimpleNet(nn.Module):
    def __init__(self, dimension, neurons_list, seed=None):
        super(SimpleNet, self).__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.layers = nn.ModuleList()  # To store all layers
        self.layers.append(nn.Linear(dimension, neurons_list[0]))
        # Hidden layers
        for i in range(1, len(neurons_list)):
            self.layers.append(nn.Linear(neurons_list[i - 1], neurons_list[i]))
        # Output layer
        self.output_layer = nn.Linear(neurons_list[-1], dimension)

    def forward(self, x):
        for layer in self.layers:
            x = F.gelu(layer(x))  # Apply GELU activation after each hidden layer
        x = self.output_layer(x)  # Output layer (no activation function)
        return x

class GradNet(nn.Module):
    def __init__(self, dimension, neurons_list, seed=None):
        super(GradNet, self).__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.layers = nn.ModuleList()  # To store all layers
        self.layers.append(nn.Linear(dimension, neurons_list[0]))
        # Hidden layers
        for i in range(1, len(neurons_list)):
            self.layers.append(nn.Linear(neurons_list[i - 1], neurons_list[i]))
        # Output layer
        self.output_layer = nn.Linear(neurons_list[-1], 1)

    def forward(self, x):
        for layer in self.layers:
            x = F.gelu(layer(x))  # Apply GELU activation after each hidden layer
        x = self.output_layer(x)  # Output layer (no activation function)
        return x