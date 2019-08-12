from .deeplab import DeepLabDecoder
from .fpn import FPNDecoder
from .linknet import LinknetDecoder
from .pspnet import PSPDecoder
from .unet import UnetDecoder


def get_decoder_cls(decoder_name):

    map =  {
        'FPN': FPNDecoder,
        'UNET': UnetDecoder,
        'LINK': LinknetDecoder,
        'PSP': PSPDecoder,
        'DEEPLAB': DeepLabDecoder,
    }

    return map[decoder_name.upper()]
