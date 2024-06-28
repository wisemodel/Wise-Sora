from .videovae import videobase_ae, videobase_ae_stride, videobase_ae_channel
from .videovae import (
    VQVAEConfiguration,
    VQVAEModel,
    VQVAETrainer,
    CausalVQVAEModel,
    CausalVQVAEConfiguration,
    CausalVQVAETrainer
)

ae_stride_config = {}
ae_stride_config.update(videobase_ae_stride)

ae_channel_config = {}
ae_channel_config.update(videobase_ae_channel)

def getae(args):
    """deprecation"""
    ae = videobase_ae.get(args.ae, None)
    assert ae is not None
    return ae(args.ae)

def getae_wrapper(ae):
    """deprecation"""
    ae = videobase_ae.get(ae, None)
    assert ae is not None
    return ae