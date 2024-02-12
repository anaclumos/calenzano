import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


class MLP(nn.Module):
    def __init__(self, num_classes=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 512)  # Input layer
        self.fc2 = nn.Linear(512, 256)  # Hidden layer
        self.fc3 = nn.Linear(256, 128)  # Hidden layer
        self.fc4 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)  # Flatten the image
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
