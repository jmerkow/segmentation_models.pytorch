import torch
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


class ExtraScalarInputsClassifier(nn.Module):

    def __init__(self, input_shape, num_hidden=None, classes=1, **input_blocks):
        input_blocks = dict(input_blocks)

        super().__init__()

        self.required_inputs = list(input_blocks.keys())
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = Flatten()

        self.module_list = nn.ModuleDict()
        total_input = input_shape
        for key, nh in input_blocks.items():
            self.module_list[key] = nn.Linear(1, nh)
            total_input += nh

        if num_hidden is not None:
            self.classifier = nn.Sequential(
                Flatten(),
                nn.Linear(total_input, num_hidden),
                nn.ReLU(),
                nn.Linear(num_hidden, classes)
            )
        else:
            self.classifier = nn.Sequential(
                Flatten(),
                nn.Linear(total_input, classes))

    def forward(self, encoder_input, **inputs):
        inputs = {k: v for k, v in inputs.items() if v is not None}
        assert inputs.keys() == self.module_list.keys(), 'incorrect keys input into network'
        outputs = [self.pool(encoder_input)]
        for k, fc in self.module_list.items():
            h = self.flatten(fc(inputs[k]))
            h = torch.unsqueeze(torch.unsqueeze(h, -1), -1)
            outputs.append(h)
        return self.classifier(torch.cat(outputs, dim=1))


classifier_map = {
    'basic': BasicClassifier,
    'latent': LatentLayerClassifier,
    'extra_scalars': ExtraScalarInputsClassifier,
}
