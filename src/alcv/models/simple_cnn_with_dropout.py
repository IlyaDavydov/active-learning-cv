import torch.nn as nn

from alcv.models.simple_cnn import SimpleCNN


class SimpleCNNWithDropout(SimpleCNN):
    """The existing SimpleCNN backbone with dropout before the final layer."""

    def __init__(self, dropout=0.3):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, images):
        outputs, _ = self.get_outputs_and_embeddings(images)
        return outputs

    def get_outputs_and_embeddings(self, images):
        embeddings = self.get_embeddings(images)
        outputs = self.linear2(self.dropout(embeddings))
        return outputs, embeddings

    def get_learning_loss_features(self, images):
        features1, features2, features3, _ = super().get_learning_loss_features(images)
        outputs = self.linear2(self.dropout(features3))
        return features1, features2, features3, outputs
