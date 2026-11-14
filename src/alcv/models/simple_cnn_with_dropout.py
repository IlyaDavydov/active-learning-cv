import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNNWithDropout(nn.Module):
    """
    Простая сверточная нейронная сеть для классификации изображений CIFAR-10.

    Модель принимает batch изображений формы:

        [batch_size, 3, 32, 32]

    где:
        batch_size — количество изображений в одном batch;
        3 — количество цветовых каналов RGB;
        32 — высота изображения;
        32 — ширина изображения.

    Модель возвращает logits формы:

        [batch_size, 10]

    где 10 — количество классов в CIFAR-10.

    Архитектура модели:

        Conv2d(3 -> 18, kernel_size=5)
        ReLU
        MaxPool2d(kernel_size=2, stride=2)

        Conv2d(18 -> 36, kernel_size=5)
        ReLU
        MaxPool2d(kernel_size=2, stride=2)

        Flatten

        Linear(900 -> 100)
        ReLU
        Linear(100 -> 10)

    Изменение формы данных:

        Input:      [batch_size, 3, 32, 32]
        Conv1:      [batch_size, 18, 28, 28]
        Pool1:      [batch_size, 18, 14, 14]
        Conv2:      [batch_size, 36, 10, 10]
        Pool2:      [batch_size, 36, 5, 5]
        Flatten:    [batch_size, 900]
        Linear1:    [batch_size, 100]
        Output:     [batch_size, 10]
    """

    def __init__(self, dropout : float) -> None:
        """
        Инициализирует слои модели.

        Здесь создаются обучаемые слои:
            conv1 — первый сверточный слой;
            conv2 — второй сверточный слой;
            linear1 — первый полносвязный слой;
            linear2 — выходной полносвязный слой.

        ReLU, MaxPool и Flatten используются в forward напрямую через
        torch.nn.functional и torch.flatten.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(3, 18, 5)
        self.conv2 = nn.Conv2d(18, 36, 5)
        self.linear1 = nn.Linear(900, 100)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(100, 10)

    def forward(self, input):
        """
        Выполняет forward pass модели.

        Аргументы:
            input:
                Tensor с изображениями формы [batch_size, 3, 32, 32].

        Возвращает:
            Tensor формы [batch_size, 10].

            Это не вероятности, а logits — сырые выходные значения модели
            для каждого из 10 классов CIFAR-10. Позже CrossEntropyLoss сам
            применит к ним нужное преобразование внутри loss-функции.
        """
        #print(f'input.shape = {input.shape} \n')
        step1 = self.conv1(input)
        #print(f'after step1.shape = {step1.shape} \n')
        step2 = F.relu(step1)
        #print(f'after step2.shape = {step2.shape} \n')
        step3 = F.max_pool2d(step2, kernel_size=2, stride=2)
        #print(f'after step3.shape = {step3.shape} \n')

        step4 = self.conv2(step3)
        #print(f'after step4.shape = {step4.shape} \n')
        step5 = F.relu(step4)
        #print(f'after step5.shape = {step5.shape} \n')
        step6 = F.max_pool2d(step5, kernel_size=2, stride=2)
        #print(f'after step6.shape = {step6.shape} \n')

        step7 = torch.flatten(step6, 1)
        #print(f'after step7.shape = {step7.shape} \n')

        step8 = self.linear1(step7)
        #print(f'after step8.shape = {step8.shape} \n')
        step9 = F.relu(step8)
        #print(f'after step9.shape = {step9.shape} \n')
        step10 = self.dropout(step9)

        output = self.linear2(step10)
        #print(f'after output.shape = {output.shape} \n')
        return output
