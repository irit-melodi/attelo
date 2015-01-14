'''
Created on Jun 27, 2012

@author: stergos


'''

from collections import defaultdict
from math import log
import sys

from depparse.graph import Digraph
from .interface import Decoder

# pylint: disable=too-few-public-methods


def get_root(_):
    """ This is used for the construction of the graph. Since we are
    using the Chu-Liu Edmonds algorithm in it's dependency parsing
    incarnation, there should be a node that is the root, i.e.  a node
    that has no incoming edges. This function is supposed to return
    that node. For the moment it always returns the first
    node. Syntactic analysis should be sufficient to provide the real
    head node.
    """
    return '1'


def _graph(instances, root='1', use_prob=True):
    """ instances are quadruplets of the form:

            source, target, probability_of_attachment, relation

        root is the "root" of the graph, that is the node that has no incoming
        nodes

        returns the Maximum Spanning Tree
    """

    targets = defaultdict(list)
    labels = dict()
    scores = dict()

    for source, target, prob, rel in instances:
        src = source.id
        tgt = target.id
        if tgt == root:
            continue
        scores[src, tgt] = prob
        labels[src, tgt] = rel
        if use_prob:  # probability scores
            scores[src, tgt] = log(prob if prob != 0.0 else sys.float_info.min)
        targets[src].append(tgt)

    return Digraph(targets,
                   lambda s, t: scores[s, t],
                   lambda s, t: labels[s, t]).mst()


def _list_edges(instances, root='1', use_prob=True):
    """ instances are quadruplets of the form:

            source, target, probability_of_attachment, relation

        root is the "root" of the graph, that is the node that has no incoming
        nodes

        returns a list of edges for the MST graph
    """
    mst = _graph(instances, root, use_prob=use_prob)

    return [(src, tgt, mst.get_label(src, tgt))
            for src, tgt in mst.iteredges()]


class MstDecoder(Decoder):
    """ attach in such a way that the resulting subgraph is a
    maximum spanning tree of the original
    """
    def __init__(self, root='1', use_prob=True):
        self._root = root
        self._use_prob = use_prob

    def decode(self, instances):
        prediction = _list_edges(instances, self._root, self._use_prob)
        return [prediction]
