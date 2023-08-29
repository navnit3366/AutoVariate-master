import torch
from torch import nn
from torch.nn import functional as F


class AutoVariate(nn.Module):
    def __init__(self, dimensions, hidden_dimensions, z_dimensions):
        super().__init__()
        self.hidden_to_input = nn.Linear(dimensions, hidden_dimensions)
        self.hidden_convert_mean = nn.Linear(hidden_dimensions, z_dimensions)
        self.hidden_convert_variance = nn.Linear(hidden_dimensions, z_dimensions)

        self.z_to_hidden = nn.Linear(z_dimensions, hidden_dimensions)
        self.hidden_to_output = nn.Linear(hidden_dimensions, dimensions)
        self.relu = nn.ReLU()

    def encode(self, x):
        hidden = self.relu(self.hidden_to_input(x))
        mean, variance = self.hidden_convert_mean(hidden), self.hidden_convert_variance(hidden)

        return mean, variance
    
    def decode(self, z):
        hidden = self.relu(self.z_to_hidden(z))
        output = torch.sigmoid(self.hidden_to_output(hidden))
        return output

    def forward(self, x):
        mean = self.encode(x)
        variance = self.encode(x)
        epsilon = torch.rand_like(variance)
        z_reparametrized = mean + variance * epsilon
        x_encoder = self.decode(z_reparametrized)
        return x_encoder, mean, variance

        