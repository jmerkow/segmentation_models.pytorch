import torch
import torch.nn as nn

from segmentation_models.encoder_decoder.base import EncoderDecoder
from segmentation_models.encoder_decoder.utils import Flatten, classifier_map
from ..decoders import get_decoder_cls
from ..encoders import get_encoder, get_preprocessing_fn


class SegmentationModel(EncoderDecoder):
    decoder_cls = None
    decoder_defaults = {}
    name = "no-decodder-{}-{}"

    def __init__(self, encoder='resnet34', activation='sigmoid',
                 encoder_weights="imagenet", classes=1,
                 encoder_classify=False, model_dir=None,
                 encoder_classifier_params=None,
                 decoder_params=None):

        decoder_params = decoder_params or {}
        encoder_classifier_params = encoder_classifier_params or {}

        self.classes = classes
        encoder_name = encoder
        encoder = get_encoder(
            encoder,
            encoder_weights=encoder_weights,
            model_dir=model_dir
        )

        defaults = self.decoder_defaults.copy()
        defaults.update(**decoder_params)
        decoder = self.decoder_cls(
            encoder_channels=encoder.out_shapes,
            final_channels=classes,
            **defaults
        )
        super().__init__(encoder=encoder, decoder=decoder, activation=activation)

        self.name = self.name.format(encoder_name)
        self.encoder_classify = encoder_classify
        self.encoder_classifier = None
        self.required_inputs = []
        if self.encoder_classify:
            classifier_type = encoder_classifier_params.pop('classifier_type', None)
            if classifier_type is None:
                self.encoder_classifier = nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    Flatten(),
                    nn.Linear(encoder.out_shapes[0], self.classes),
                )
            else:
                klass = classifier_map[classifier_type]
                self.encoder_classifier = klass(input_shape=encoder.out_shapes[0],
                                                classes=self.classes, **encoder_classifier_params)

            self.required_inputs = getattr(self.encoder_classifier, "required_inputs", [])
            self.name += "-wclassifier"

    def forward(self, x, **args):
        """Sequentially pass `x` trough model`s `encoder` and `decoder` (return logits!)"""
        features = self.encoder(x)
        mask = self.decoder(features)
        if self.encoder_classify:
            scalars = {k: args[k] for k in self.required_inputs}
            score = self.encoder_classifier(features[0], **scalars)
            return [mask, score]
        return mask

    def predict(self, x, **args):
        """Inference method. Switch model to `eval` mode, call `.forward(x)`
        and apply activation function (if activation is not `None`) with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        with torch.no_grad():
            mask = self.forward(x, **args)
            score = None
            if self.encoder_classify:
                mask, score = mask
            if self.activation:
                mask = self.activation(mask)
                if score is not None:
                    score = self.activation(score)
            if self.encoder_classify:
                x = [mask, score]
            else:
                x = mask

        return x

    def get_encoder_params(self):
        return self.encoder.parameters()

    def get_decoder_params(self):
        return self.decoder.parameters()

    def get_classifier_params(self):
        return self.encoder_classifier.parameters()


class UNet(SegmentationModel):
    decoder_cls = get_decoder_cls('UNET')
    decoder_defaults = {'use_batchnorm': True,
                        'decoder_channels': (256, 128, 64, 32, 16),
                        'center': False}
    name = 'unet-{}'


class PSPNet(SegmentationModel):
    decoder_cls = get_decoder_cls('PSP')
    decoder_defaults = {'downsample_factor': 8,
                        'psp_out_channels': 512,
                        'use_batchnorm': True,
                        'aux_output': False,
                        'dropout': 0.2}
    name = 'psp-{}'


class LinkNet(SegmentationModel):
    decoder_cls = get_decoder_cls('LINK')
    decoder_defaults = {'prefinal_channels': 32}
    name = 'link-{}'


class FPN(SegmentationModel):
    decoder_cls = get_decoder_cls('FPN')
    decoder_defaults = {'pyramid_channels': 256, 'segmentation_channels': 128, 'dropout': 0.2}
    name = 'fpn-{}'


class VDeepLab(SegmentationModel):
    decoder_cls = get_decoder_cls('DEEPLAB')
    decoder_defaults = {"decoder_channels": (256, 48), "output_stride": 8}
    name = 'vdeeplab-{}'


class UNetPP(SegmentationModel):
    decoder_defaults = {'decoder_channels': (256, 128, 64, 32, 16), 'upsample_mode': 'bilinear', 'dropout': 0}
    name = 'unetpp-{}'
    decoder_cls = get_decoder_cls('UNETPP')


models_types = {
    'unet': UNet,
    'pspnet': PSPNet,
    'linknet': LinkNet,
    'fpn': FPN,
    'vdeeplab': VDeepLab,
    'unetpp': UNetPP,
}


def get_model(base, **model_params):
    base = base.lower()
    model_cls = models_types[base]
    model = model_cls(**model_params)

    encoder_weights = model_params['encoder_weights']
    preprocessing = None
    if encoder_weights is not None:
        preprocessing = get_preprocessing_fn(model_params['encoder'], pretrained=encoder_weights)

    return model, preprocessing
