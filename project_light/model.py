import torch.nn as nn
from torchvision import models


def create_model(num_classes=3, pretrained=True):
    """
    ResNet18 с кастомной головой для классификации уровня освещённости.
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)

    # Замораживаем все слои
    for param in model.parameters():
        param.requires_grad = False

    # Размораживаем только layer4 (последний блок)
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Кастомная голова
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    return model


CLASS_NAMES = {0: 'dark', 1: 'normal', 2: 'bright'}
CLASS_NAMES_RU = {0: 'Тёмно', 1: 'Нормально', 2: 'Ярко'}
CLASS_EMOJI = {0: '🌑', 1: '🌤️', 2: '☀️'}
CLASS_COLOR = {0: '#2c3e50', 1: '#f39c12', 2: '#f1c40f'}