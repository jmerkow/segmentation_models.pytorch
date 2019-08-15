from .deeplab import DeepLabDecoder
from .fpn import FPNDecoder
from .linknet import LinknetDecoder
from .pspnet import PSPDecoder
from .unet import UnetDecoder
from .unetpp import UNetPPDecoder


def get_decoder_cls(decoder_name):

    map =  {
        'FPN': FPNDecoder,
        'UNET': UnetDecoder,
        'LINK': LinknetDecoder,
        'PSP': PSPDecoder,
        'DEEPLAB': DeepLabDecoder,
        'UNETPP': UNetPPDecoder,
    }

    return map[decoder_name.upper()]
