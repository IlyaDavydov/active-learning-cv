import torch.nn.functional as F
from torch import nn


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(32, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, 1)

    def forward(self, x):
        step1 = F.relu(self.linear1(x))
        step2 = F.relu(self.linear2(step1))
        output = F.sigmoid(self.linear3(step2))

        return output

