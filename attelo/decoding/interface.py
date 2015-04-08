'''
Common interface that all decoders must implement
'''

from __future__ import print_function
from abc import ABCMeta, abstractmethod
from six import with_metaclass

# pylint: disable=abstract-class-not-used, abstract-class-little-used
# pylint: disable=too-few-public-methods


class Decoder(with_metaclass(ABCMeta, object)):
    '''
    A decoder is a function which given a probability distribution (see below)
    and some control parameters, returns a sequence of predictions.

    Most decoders only really return one prediction in practice, but some,
    like the A* decoder might have able to return a ranked sequence of
    the "N best" predictions it can find

    We have a few informal types to consider here:

        - a **link** (`(string, string, string)`) represents a link
          between a pair of EDUs. The first two items are their
          identifiers, and the third is the link label

        - a **candidate link** (or candidate, to be short,
          `(string, string, float, string)`)
          is a link with a probability attached

        - a **prediction** is morally a set (in practice a list) of links

        - a **distribution** is morally a set of proposed links
    '''

    @abstractmethod
    def decode(self, candidates):
        '''
        :param candidates: the proposed links that we would like
                             to decode over
        :type candidates: [(string, string, float, string)]

        The links you return must be a subset of the proposed links
        from the probability distribution (modulo probabilities).
        They can be in any order.

        :rtype: [ [(string,string,string)] ]
        '''
        raise NotImplementedError


class PruningDecoder(with_metaclass(ABCMeta, Decoder)):
    '''
    A pruning decoder takes another decoder as input and does some
    preprocessing on the candidate edges, removing some of them
    before handing them off to its inner decoder
    '''
    def __init__(self, decoder):
        self.decoder = decoder

    def decode(self, candidates):
        return self.decoder.decode(self.prune(candidates))

    @abstractmethod
    def prune(self, candidates):
        '''
        :param candidates: the proposed links that we would like
                           to decode over
        :type candidates: [(string, string, float, string)]

        Trim a set of proposed links

        :rtype: [(string, string, float, string)]
        '''
        raise NotImplementedError
