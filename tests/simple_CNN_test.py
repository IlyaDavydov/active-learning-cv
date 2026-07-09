from alcv.models.simple_cnn import SimpleCNN
import torch

def test_simple_cnn_forward_output_shape():
    """
    Проверяет, что SimpleCNN принимает одно CIFAR-10 изображение
    формы [1, 3, 32, 32] и возвращает logits формы [1, 10].
    """
    model = SimpleCNN()

    rand_arr = torch.rand(1, 3, 32, 32)
    out = model(rand_arr)

    assert out.shape == torch.Size([1, 10])

def test_simple_cnn_forward_batch_output_shape():
    """
    Проверяет, что SimpleCNN корректно работает не только с одним изображением,
    но и с batch из нескольких изображений.

    Вход:
        [8, 3, 32, 32]

    Ожидаемый выход:
        [8, 10]
    """
    model = SimpleCNN()

    rand_arr = torch.rand(8, 3, 32, 32)
    out = model(rand_arr)

    assert out.shape == torch.Size([8, 10])