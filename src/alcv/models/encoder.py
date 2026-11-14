import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=4, stride=2, padding=1) # [N, 3, 32, 32] -> [N, 128, 16, 16]
        self.conv2 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1) # [N, 128, 16, 16] -> [N, 256, 8, 8]
        self.conv3 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1) # [N, 256, 8, 8] -> [N, 512, 4, 4]
        self.conv4 = nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1) # [N, 512, 4, 4] -> [N, 1024, 2, 2]
        self.linear_mu =nn.Linear(4096, 32)
        self.linear_logvar = nn.Linear(4096, 32)

    def forward(self, x):
        step1 = F.relu(self.conv1(x))
        step2 = F.relu(self.conv2(step1))
        step3 = F.relu(self.conv3(step2))
        step4 = F.relu(self.conv4(step3))

        features = torch.flatten(step4, 1) # [N, 4096]
        mu = self.linear_mu(features) # [N, 32]
        logvar = self.linear_logvar(features) # [N, 32]

        return mu, logvar

    def parametrize(self, mu, logvar):
        epsilon = torch.randn_like(mu)
        std = torch.exp(0.5 * logvar)
        z = mu + std * epsilon
        return z