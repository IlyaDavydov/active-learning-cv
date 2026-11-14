import torch
import torch.nn as nn
import torch.nn.functional as F


class LossPredictionModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_features1 = nn.Linear(18, 128)
        self.linear_features2 = nn.Linear(36, 128)
        self.linear_features3 = nn.Linear(100, 128)
        self.linear_output = nn.Linear(384, 1)

    def forward(self, features1, features2, features3):
        features1 = torch.mean(features1, dim=(2, 3))
        features1 = self.linear_features1(features1)
        features1 = F.relu(features1)

        features2 = torch.mean(features2, dim=(2, 3))
        features2 = self.linear_features2(features2)
        features2 = F.relu(features2)

        features3 = self.linear_features3(features3)
        features3 = F.relu(features3)

        features = torch.cat((features1, features2, features3), dim=1)

        predicted_loss = self.linear_output(features)

        return predicted_loss