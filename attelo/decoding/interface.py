'''
Common interface that all decoders must implement
'''

from __future__ import print_function
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from six import with_metaclass

import numpy as np

# pylint: disable=too-few-public-methods

class LinkPack(namedtuple('LinkPack',
                          ['edus',
                           'pairings',
                           'labels',
                           'scores_ad',
                           'scores_l'])):
    '''
    Collection of candidate links

    :param labels: list of labels (same length as the width of
                   scores_l), nb. may not match DataPack labels
    :type labels: [string]

    :param edus: list of EDUs (somewhat redundant with pairings,
                 but provided so we resemble DataPack)
    :type edus: [(EDU, EDU)]

    :param pairings: list of EDU pairs (length = number of samples)
    :type pairings: [(EDU, EDU)]

    :param scores_ad: directed attachment scores (length should
                      match that of pairings)
    :type scores_ad: array(float)

    :param scores_l: label attachment scores (width should be
                     number of pairings, height should be number
                     of labels)
    :type scores_l: array(float)
    '''
    def __len__(self):
        return len(self.pairings)

    def selected(self, indices):
        '''
        Return a subset of the links indicated by the list/array
        of indices
        '''
        sel_pairings = [self.pairings[x] for x in indices]
        sel_edus_ = set()
        for edu1, edu2 in sel_pairings:
            sel_edus_.add(edu1)
            sel_edus_.add(edu2)
        sel_edus = [e for e in self.edus if e in sel_edus_]
        sel_scores_ad = self.scores_ad[indices]
        return LinkPack(edus=sel_edus,
                        labels=self.labels,
                        pairings=sel_pairings,
                        scores_ad=sel_scores_ad,
                        scores_l=self.scores_l[indices])
        # pylint: enable=no-member

    def simple_candidates(self):
        '''
        Translate the links into a list of (EDU, EDU, float, string)
        quadruplets representing a combined probability and best label
        for each EDU pair.  This is often good enough for simplistic
        decoders
        '''
        # pylint: disable=no-member
        scores_best_l = np.ravel(np.amax(self.scores_l, axis=1))
        best_lbls = np.ravel(np.argmax(self.scores_l, axis=1))
        scores = np.multiply(self.scores_ad, scores_best_l)
        # pylint: enable=no-member
        return [(pair[0], pair[1], score, self.get_label(lbl))
                for pair, score, lbl
                in zip(self.pairings, scores, best_lbls)]

    def get_label(self, i):
        '''
        Return the class label for the given target value.
        '''
        return self.labels[int(i)]

    def label_number(self, label):
        '''
        Return the numerical label that corresponnds to the given
        string label

        :rtype: float
        '''
        return self.labels.index(label)


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
          `(EDU, EDU, float, string)`)
          is a link with a probability attached

        - a **prediction** is morally a set (in practice a list) of links

        - a **distribution** is morally a set of proposed links
    '''

    @abstractmethod
    def decode(self, lpack):
        '''
        :type lpack: LinkPack
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

    def decode(self, lpack):
        return self.decoder.decode(self.prune(lpack))

    @abstractmethod
    def prune(self, lpack):
        '''
        Trim a set of proposed links

        :rtype: LinkPack
        '''
        raise NotImplementedError
