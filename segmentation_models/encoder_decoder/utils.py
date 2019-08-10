from torch import nn as nn


class Flatten(nn.Module):
    def forward(self, input):
        return input.reshape(input.size(0), -1)


def BasicClassifier(input_shape, classes=1):
    return nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        Flatten(),
        nn.Linear(input_shape, classes),
    )


def LatentLayerClassifier(input_shape, classes=1, num_hidden=1024):
    return nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        Flatten(),
        nn.Linear(input_shape, num_hidden),
        nn.ReLU(),
        nn.Linear(num_hidden, classes)
    )


classifier_map = {
    'basic': BasicClassifier,
    'latent': LatentLayerClassifier,
}
