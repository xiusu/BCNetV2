from .base import BaseCompressor

from ..utils import CompressorReg

@CompressorReg.register_module()
class EmptyCompressor(BaseCompressor):

    def __init__(self):
        pass
