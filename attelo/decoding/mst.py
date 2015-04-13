'''
Created on Jun 27, 2012

@author: stergos, jrmyp
'''

from __future__ import print_function
from collections import defaultdict

from depparse.graph import Digraph
# pylint: disable=no-name-in-module
from scipy.special import logit
# pylint: enable=no-name-in-module

from ..edu import FAKE_ROOT_ID
from ..util import ArgparserEnum
from .interface import Decoder
from .util import DecoderException

# pylint: disable=too-few-public-methods

# see _cap_score
MAX_SCORE = 1e90
MIN_SCORE = -MAX_SCORE


def _cap_score(score):
    '''
    the depparse package's MST implementation uses a hardcoded minimum score of
    `-1e100`.
    Feeding it lower weights crashes the algorithm
    We set minimum and maximum scores to avoid this
    Unless we have more than 1e10 nodes, combined scores can't reach the limit

    :type score: float
    :rtype: float
    '''
    return min(MAX_SCORE, max(MIN_SCORE, score))


def _leftmost_edu(edus):
    """ Returns the default root node for MST/MSDAG algorithm

        Currently, the EDU in first position.

        The Chu-Liu-Edmonds algorithm used for MST/MSDAG requires a
        root node (with no incoming edges). We ensure there is one.
    """
    return sorted(edus, key=lambda e: e.span)[0]


def _msdag(graph):
    """ Returns a subgraph of graph (a Digraph) corresponding to its
        Maximum Spanning Directed Acyclic Graph

        Algorithm is semi-greedy-MSDAG as described in Schluter_:
        .. _Schluter (2014): http://aclweb.org/anthology/W14-2412
    """
    tree = graph.mst()
    # Sort edges in orginal graph by decreasing score
    edges = sorted(graph.iteredges(), key=lambda p: -graph.get_score(*p))

    for src, tgt in edges:
        # Already in graph ?
        if tgt in tree.successors[src]:
            continue
        # Add the edge, revert if cycle is created
        tree.successors[src].append(tgt)
        if tree.find_cycle():
            tree.successors[src].remove(tgt)

    # Update score and label functions
    new_map = lambda f: dict(((s, t), f(s, t)) for s, t in tree.iteredges())
    nscores = new_map(graph.get_score)
    nlabels = new_map(graph.get_label)

    return Digraph(tree.successors,
                   lambda s, t: nscores[s, t],
                   lambda s, t: nlabels[s, t])


class MstRootStrategy(ArgparserEnum):
    '''
    How we declare the MST root node
    '''
    fake_root = 1
    leftmost = 2


class MstDecoder(Decoder):
    """ Attach in such a way that the resulting subgraph is a
        maximum spanning tree of the original
    """
    def __init__(self, root_strategy, use_prob=True):
        self._use_prob = use_prob
        self._root_strategy = root_strategy

    def _graph(self, instances):
        """ Builds a directed graph for instances

            instances are quadruplets of the form:
                edu_source, edu_target, probability_of_attachment, relation

            :rtype Digraph
        """

        if self._root_strategy == MstRootStrategy.leftmost:
            root_id = _leftmost_edu(set(e for s, t, _, _ in instances
                                        for e in (s, t))).id
        elif self._root_strategy == MstRootStrategy.fake_root:
            root_id = FAKE_ROOT_ID
        else:
            raise DecoderException('Unknown root finding strategy: ' +
                                   str(self._root_strategy))

        targets = defaultdict(list)
        labels = dict()
        scores = dict()

        for source, target, prob, rel in instances:
            src, tgt = source.id, target.id

            # Ignore all edges directed to the root
            if tgt == root_id:
                continue

            if self._use_prob:
                scores[src, tgt] = _cap_score(logit(prob))
            else:
                scores[src, tgt] = prob
            labels[src, tgt] = rel
            targets[src].append(tgt)

        return Digraph(targets,
                       lambda s, t: scores[s, t],
                       lambda s, t: labels[s, t])

    def decode(self, lpack):
        graph = self._graph(lpack.simple_candidates())
        subgraph = graph.mst()
        predictions = [(src, tgt, subgraph.get_label(src, tgt))
                       for src, tgt in subgraph.iteredges()]
        return [predictions]


class MsdagDecoder(MstDecoder):
    """ Attach according to MSDAG (subgraph of original)"""

    def decode(self, lpack):
        graph = self._graph(lpack.simple_candidates())
        subgraph = _msdag(graph)
        predictions = [(src, tgt, subgraph.get_label(src, tgt))
                       for src, tgt in subgraph.iteredges()]
        return [predictions]
