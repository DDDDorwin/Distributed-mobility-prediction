import torch
import torch.nn as nn


class CNNnetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d = nn.Conv1d(1,64,kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.Linear1= nn.Linear(64*6,50)
        self.Linear2= nn.Linear(50,1)
        self.model = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64 * 6, 50),
            nn.ReLU(inplace=True),
            nn.Linear(50, 1)
        )

    def forward(self, x):
        # x = self.conv1d(x)
        # x = self.relu(x)
        # x = x.view(-1)
        # x = self.Linear1(x)
        # x = self.relu(x)
        # x = self.Linear2(x)
        output = self.model(x)
        return output
