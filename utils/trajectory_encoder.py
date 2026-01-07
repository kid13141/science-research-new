import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, feature_dim)
        self.fc2 = nn.Linear(feature_dim, feature_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class AutoregressiveModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AutoregressiveModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)

    def forward(self, x):
        _, hn = self.gru(x)  # We only need the hidden state at the last timestep
        return hn.squeeze(0)  # Remove the num_layers dimension

