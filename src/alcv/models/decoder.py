from torch import nn
import torch.nn.functional as F
import torch

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(32, 4096) # [N, 4096]
        self.conv_transponse_1 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1) # new = old * stride - 2 * padding + kernel, [N, 1024, 2, 2] -> [N, 512, 4, 4]
        self.conv_transponse_2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1) # [N, 256, 8, 8]
        self.conv_transponse_3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1) # [N, 128, 16, 16]
        self.conv_transponse_4 = nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1) # [N, 3, 32, 32]

    def forward(self, x):
        step1 = F.relu(self.linear(x))

        reshaped_tensor = torch.reshape(step1, (step1.shape[0], 1024, 2, 2))

        step2 = F.relu(self.conv_transponse_1(reshaped_tensor))
        step3 = F.relu(self.conv_transponse_2(step2))
        step4 = F.relu(self.conv_transponse_3(step3))
        output = torch.sigmoid(self.conv_transponse_4(step4))

        return output

