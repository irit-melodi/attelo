'''
Created on Jun 27, 2012

@author: stergos, jrmyp


'''

from collections import defaultdict
from math import log
from functools import wraps
import sys


from depparse.graph import Digraph
from .interface import Decoder

# pylint: disable=too-few-public-methods

class DigraphPlus(Digraph):
    """ Extenstion of Digraph, with MSDAG """
    
    def __init__(self, *a, **kw):
        """ Initialize graph """
        super(DigraphPlus, self).__init__(*a, **kw)
    
    @classmethod
    def clone(cls, dg):
        return cls(dg.successors, dg.get_score, dg.get_label)
        
    def msdag(self):
        """ Return the MSDAG of the graph using a variant of CLE
        
        Returns a new DigraphPlus
        """
        mark = Digraph.new_node_id
        # Consider all edges instead of subgraph
        candidate = self
        cycle = candidate.find_cycle()
        if not cycle:
            return candidate
        new_id, old_edges, compact = self.contract(cycle)
        merged = self.merge(compact.msdag(), new_id, old_edges, cycle)
        return merged

    def find_cycle(self, *a, **ka):
        """ Similar to Digraph.find_cycle, but returns a DigraphPlus """
        res = super(DigraphPlus, self).find_cycle(*a, **ka)
        return res if (res is None) else DigraphPlus.clone(res)

    def merge(self, *a, **ka):
        """ Similar to Digraph.merge, but returns a DigraphPlus """
        return DigraphPlus.clone(super(DigraphPlus, self).merge(*a, **ka))

    def contract(self, *a, **ka):
        """ Similar to Digraph.contract, but returns a DigraphPlus """
        new_id, old_edges, dg = super(DigraphPlus, self).contract(*a, **ka)
        return new_id, old_edges, DigraphPlus.clone(dg)

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

        returns the Graph
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
                   lambda s, t: labels[s, t])


def _list_edges(instances, root='1', use_prob=True, use_msdag=False):
    """ instances are quadruplets of the form:

            source, target, probability_of_attachment, relation

        root is the "root" of the graph, that is the node that has no incoming
        nodes

        returns a list of edges for the MST/MSDAG graph
    """
    graph = _graph(instances, root, use_prob=use_prob)
    subgraph = graph.msdag() if use_msdag else graph.mst()

    return [(src, tgt, subgraph.get_label(src, tgt))
            for src, tgt in subgraph.iteredges()]

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

    return _list_edges(*args, **kwargs)

def msdag_decoder(*args, **kwargs):
    """ Attach according to MSDAG (subgraph of original) """
    return _list_edges(*args, use_msdag=True, **kwargs)
