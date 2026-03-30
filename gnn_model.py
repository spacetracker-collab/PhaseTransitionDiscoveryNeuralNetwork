
import torch
import torch.nn as nn

class SimpleGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2,16)
        self.fc2 = nn.Linear(16,8)
        self.fc3 = nn.Linear(8,2)

    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
