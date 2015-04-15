'''
Common interface that all decoders must implement
'''

from __future__ import print_function
from abc import ABCMeta, abstractmethod
from six import with_metaclass

from attelo.parser import Parser

# pylint: disable=too-few-public-methods


class Decoder(with_metaclass(ABCMeta, Parser)):
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
          `(EDU, EDU, float, string)`)
          is a link with a probability attached

        - a **prediction** is morally a set (in practice a list) of links

        - a **distribution** is morally a set of proposed links

    Note that a decoder could also be seen/used as a sort of crude parser
    (with a fit function is a no-op). You'll likely want to prefix it with
    a parser that extracts weights from datapacks lest you work with the
    somewhat unformative 1.0s everywhere.
    '''

    @abstractmethod
    def decode(self, dpack):
        '''
        Return the N-best predictions in the form of a datapack per
        prediction.
        '''
        raise NotImplementedError

    def fit(self, dpacks, targets):
        return

    def transform(self, dpack):
        dpack = self.multiply(dpack) # default weights if not set
        return self.decode(dpack)
