from segmentation_models.encoder_decoder.base import EncoderDecoder
from ..decoders import get_decoder_cls
from ..encoders import get_encoder


class SegmentationModel(EncoderDecoder):
    decoder_cls = None
    decoder_defaults = {}
    name = "no-decodder-{}-{}"

    def __init__(self, encoder_name='resnet34', activation='sigmoid',
                 encoder_weights="imagenet", classes=1,
                 **decoder_kwargs):
        self.classes = classes
        encoder = get_encoder(
            encoder_name,
            encoder_weights=encoder_weights
        )

        defaults = self.decoder_defaults.copy()
        defaults.update(**decoder_kwargs)
        decoder = self.decoder_cls(
            encoder_channels=encoder.out_shapes,
            final_channels=classes,
            **defaults
        )
        super().__init__(encoder=encoder, decoder=decoder, activation=activation)
        self.name = self.name.format(encoder_name)


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


# def get_segmententation_model(encoder_name='resnet34', decoder_name=None, decoder_kwargs=None,
#                               activation='sigmoid',
#                               encoder_weights="imagenet", name=None, classes=1):
#     class SegModel(EncoderDecoder):
#         def __init__(
#                 self,
#                 encoder_name,
#                 decoder_name,
#                 activation='sigmoid',
#                 encoder_weights="imagenet",
#                 name=None,
#                 decoder_kwargs=None,
#                 classes=1,
#
#         ):
#             assert decoder_name is not None, "must specify decoder"
#             decoder_kwargs = decoder_kwargs or {}
#
#             encoder = get_encoder(
#                 encoder_name,
#                 encoder_weights=encoder_weights
#             )
#
#             decoder_cls = get_decoder_cls(decoder_name)
#
#             decoder = decoder_cls(
#
#                 **decoder_kwargs
#             )
#
#             self.classes = classes
#             super().__init__(encoder, decoder, activation)
#
#             if name is None:
#                 name = '{}-{}-{}'.format(decoder_name.lower(), encoder_name, activation)
#             self.name = name
#
#     return SegModel(encoder_name, decoder_name, activation=activation,
#                     encoder_weights=encoder_weights,
#                     name=name, classes=classes,
#                     decoder_kwargs=decoder_kwargs)
#
#
# def UNet(**kwargs):
#     decoder_kwargs = dict(
#         use_batchnorm=True,
#         decoder_channels=(256, 128, 64, 32, 16),
#         center=False)
#     decoder_kwargs.update(kwargs.pop('decoder_kwargs', {}) or {})
#     kwargs['decoder_name'] = 'UNET'
#     kwargs['decoder_kwargs'] = decoder_kwargs
#     kwargs.setdefault('activation', 'sigmoid')
#
#     return get_segmententation_model(**kwargs)
#
#
# def PSPNet(**kwargs):
#     decoder_kwargs = dict(
#         downsample_factor=8,
#         psp_out_channels=512,
#         use_batchnorm=True,
#         aux_output=False,
#         dropout=0.2, )
#     decoder_kwargs.update(kwargs.pop('decoder_kwargs', {}) or {})
#     kwargs['decoder_name'] = 'PSP'
#     kwargs['decoder_kwargs'] = decoder_kwargs
#     kwargs.setdefault('activation', 'sigmoid')
#
#     return get_segmententation_model(**kwargs)
#
#
# def LinkNet(**kwargs):
#     decoder_kwargs = dict(
#         prefinal_channels=32,
#     )
#     decoder_kwargs.update(kwargs.pop('decoder_kwargs', {}) or {})
#     kwargs['decoder_name'] = 'LINK'
#     kwargs['decoder_kwargs'] = decoder_kwargs
#     kwargs.setdefault('activation', 'sigmoid')
#
#     return get_segmententation_model(**kwargs)
#
#
# def FPN(**kwargs):
#     decoder_kwargs = dict(
#             pyramid_channels=256,
#             segmentation_channels=128,
#             dropout=0.2
#     )
#     decoder_kwargs.update(kwargs.pop('decoder_kwargs', {}) or {})
#     kwargs['decoder_name'] = 'FPN'
#     kwargs['decoder_kwargs'] = decoder_kwargs
#     kwargs.setdefault('activation', 'sigmoid')
#
#     return get_segmententation_model(**kwargs)
