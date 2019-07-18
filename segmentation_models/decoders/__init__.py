from .fpn import FPNDecoder
from .unet import UnetDecoder
from .linknet import LinknetDecoder
from .pspnet import PSPDecoder


def get_decoder_cls(decoder_name):

    map =  {
        'FPN': FPNDecoder,
        'UNET': UnetDecoder,
        'LINK': LinknetDecoder,
        'PSP': PSPDecoder,
    }

    return map[decoder_name.upper()]
