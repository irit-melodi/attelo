'''
A "pruning" decoder that pre-processes candidate edges and prunes them away
if they are separated by more than a certain number of EDUs
'''
# pylint: disable=too-few-public-methods

from .interface import (PruningDecoder)
from ..table import (select_window)


class WindowPruningDecoder(PruningDecoder):
    '''
    Note that we assume that the probability distribution includes every
    EDU in its grouping.

    If there are any gaps, the window will be a bit messed up
    '''
    def __init__(self, decoder, window):
        super(WindowPruningDecoder, self).__init__(decoder)
        self._window = window

    def prune(self, lpack):
        # select_window will work for LinkPacks via duck typing
        return select_window(lpack, self._window)
