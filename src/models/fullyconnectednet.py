import torch 
from torch import nn

class FullConnectedNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer = nn.Sequential(
            nn.Linear(128 * 128 * 3 , 128), # 128 * 128 = 16 484
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, 1024), 
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024 , 2048), 
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(2048, 128 * 128 * 3),
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.layer(x)
        x = x.view(-1, 3, 128, 128)  
        return x
