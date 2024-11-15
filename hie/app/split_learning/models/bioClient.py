import torch.nn as nn
import torch.nn.functional as F


class InputEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(8, 32)

    def forward(self, x):
        # print(x.shape, x)
        x = F.relu(self.lin1(x))
        return x


class OutputDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.lin3(x))
        # print(x.shape, x)
        return x
