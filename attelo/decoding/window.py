'''
A "pruning" decoder that pre-processes candidate edges and prunes them away
if they are separated by more than a certain number of EDUs
'''
# pylint: disable=too-few-public-methods

from .interface import (Decoder)
from attelo.table import (select_window)


class WindowPruner(Decoder):
    '''
    Notes
    -----
    We assume that the datapack includes every EDU in its
    grouping.

    If there are any gaps, the window will be a bit messed up

    As decoders are parsers like any other, if you just want
    to apply this as preprocessing to a decoder, you could
    construct a mini pipeline consisting of this plus the
    decoder. Alternatively, if you already have a larger
    pipeline of which the decoder is already part, you can
    just insert this before the decoder.
    '''
    def __init__(self, window):
        super(WindowPruner, self).__init__()
        self._window = window

    def decode(self, dpack):
        return select_window(dpack, self._window)
