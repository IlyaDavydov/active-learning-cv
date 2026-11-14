import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 18, 5)
        self.conv2 = nn.Conv2d(18, 36, 5)
        self.linear1 = nn.Linear(900, 100)
        self.linear2 = nn.Linear(100, 10)

    def forward(self, input):
        step1 = self.conv1(input)
        step2 = F.relu(step1)
        step3 = F.max_pool2d(step2, kernel_size=2, stride=2)

        step4 = self.conv2(step3)
        step5 = F.relu(step4)
        step6 = F.max_pool2d(step5, kernel_size=2, stride=2)

        step7 = torch.flatten(step6, 1)

        step8 = self.linear1(step7)
        step9 = F.relu(step8)

        output = self.linear2(step9)

        return output

    def get_embeddings(self, input):
        step1 = self.conv1(input)
        step2 = F.relu(step1)
        step3 = F.max_pool2d(step2, kernel_size=2, stride=2)

        step4 = self.conv2(step3)
        step5 = F.relu(step4)
        step6 = F.max_pool2d(step5, kernel_size=2, stride=2)

        step7 = torch.flatten(step6, 1)

        step8 = self.linear1(step7)
        embeddings = F.relu(step8)

        return embeddings

    def get_outputs_and_embeddings(self, input):
        step1 = self.conv1(input)
        step2 = F.relu(step1)
        step3 = F.max_pool2d(step2, kernel_size=2, stride=2)

        step4 = self.conv2(step3)
        step5 = F.relu(step4)
        step6 = F.max_pool2d(step5, kernel_size=2, stride=2)

        step7 = torch.flatten(step6, 1)

        step8 = self.linear1(step7)
        embeddings = F.relu(step8)

        outputs = self.linear2(embeddings)

        return outputs, embeddings

    def get_learning_loss_features(self, input):
        step1 = self.conv1(input)
        step2 = F.relu(step1)
        step3 = F.max_pool2d(step2, kernel_size=2, stride=2)

        step4 = self.conv2(step3)
        step5 = F.relu(step4)
        step6 = F.max_pool2d(step5, kernel_size=2, stride=2)

        step7 = torch.flatten(step6, 1)

        step8 = self.linear1(step7)
        step9 = F.relu(step8)

        output = self.linear2(step9)

        return step3, step6, step9, output