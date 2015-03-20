'''
A "pruning" decoder that pre-processes candidate edges and prunes them away
if they are separated by more than a certain number of EDUs
'''
# pylint: disable=too-few-public-methods

from .interface import (PruningDecoder)
from .util import (get_sorted_edus)


class WindowPruningDecoder(PruningDecoder):
    '''
    Note that we assume that the probability distribution includes every
    EDU in its grouping.

    If there are any gaps, the window will be a bit messed up
    '''
    def __init__(self, decoder, window):
        super(WindowPruningDecoder, self).__init__(decoder)
        self._window = window

    def prune(self, prob_distrib):
        positions = {e: i for i, e in enumerate(get_sorted_edus(prob_distrib))}
        prob_distrib2 = []
        for inst in prob_distrib:
            edu1, edu2, _, _ = inst
            gap = abs(positions[edu2] - positions[edu1])
            if gap <= self._window:
                prob_distrib2.append(inst)
        return prob_distrib2
