"""
План:

функция получает:
- model — обученную модель
- unlabeled_loader — DataLoader с изображениями, которые пока считаются неразмеченными
- sample_size — количество изображений, которые нужно выбрать

Перед этим:
- исходный датасет оборачивается в DatasetWithIndices
- благодаря этому DataLoader возвращает не только изображение и метку,
  но также настоящий индекс изображения в исходном датасете

В процессе:
- модель получает каждое изображение из unlabeled_loader
- для каждого изображения модель выдаёт logits для всех классов
- logits преобразуются в вероятности с помощью softmax
- находится самая большая вероятность среди классов
- считается score:

    score = 1 - максимальная вероятность

Например:

изображение #1:
[0.6 собака, 0.3 кошка, 0.1 кит]

максимальная вероятность = 0.6
score = 1 - 0.6 = 0.4

изображение #2:
[0.2 собака, 0.7 кошка, 0.1 кит]

максимальная вероятность = 0.7
score = 1 - 0.7 = 0.3

Так как 0.4 > 0.3, модель меньше уверена в изображении #1.
Поэтому изображение #1 предпочтительнее выбрать для дальнейшей разметки и обучения.

Для каждого изображения сохраняются:
- его настоящий индекс в исходном датасете
- его least confidence score

После обработки всех изображений:
- пары индекс-score сортируются по score от большего к меньшему
- выбираются первые sample_size индексов

Функция возвращает:
- selected_indices — список индексов изображений,
  в которых модель уверена меньше всего
"""

import torch
from torch.utils.data import Dataset


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


class DatasetWithIndices(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]

        return image, label, index


def least_confidence(model, unlabeled_loader, sample_size):
    model = model.to(device)
    model.eval()

    indices_and_scores = []

    with torch.no_grad():
        for images, _, indices in unlabeled_loader:
            images = images.to(device)

            outputs = model(images)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            max_probabilities = probabilities.max(dim=1).values

            scores = 1 - max_probabilities

            for index, score in zip(indices, scores):
                indices_and_scores.append(
                    (index.item(), score.item())
                )

    indices_and_scores.sort(
        key=lambda item: item[1],
        reverse=True
    )

    selected_indices = [
        index
        for index, score
        in indices_and_scores[:sample_size]
    ]

    return selected_indices