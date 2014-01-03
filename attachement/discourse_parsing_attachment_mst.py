'''
Created on Jun 27, 2012

@author: stergos


'''


from depparse.graph import Digraph
from math import log
import sys


def get_root(data):
    """ This is used for the construction of the graph. Since we are
    using the Chu-Liu Edmonds algorithm in it's dependency parsing
    incarnation, there should be a node that is the root, i.e.  a node
    that has no incoming edges. This function is supposed to return
    that node. For the moment it always returns the first
    node. Syntactic analysis should be sufficient to provide the real
    head node.
    """
    return '1'

def MST_graph(instances, root = '1', use_prob=True):
    """ instances are quadruplets of the form:

            source, target, probability_of_attachment, relation

        root is the "root" of the graph, that is the node that has no incoming nodes

        returns the Maximum Spanning Tree
    """

    map = dict()
    scores = dict()

    for source, target, p, r in instances :
        s = source.id
        t = target.id
        if t == root:
            continue
        scores[s, t] = p
        if use_prob: # probability scores
            if p == 0.0:
                scores[s, t] = log(sys.float_info.min) 
            else :
                scores[s, t] = log(p)
        if s in map :
            map[s].append(t)
        else :
            map[s] = [t]

    return Digraph(map, lambda s, t: scores[s, t], lambda s, t: r).mst()

def MST_list_edges(instances, root = '1', use_prob=True):
    """ instances are quadruplets of the form:

            source, target, probability_of_attachment, relation

        root is the "root" of the graph, that is the node that has no incoming nodes

        returns a list of edges for the MST graph
    """
    mst = MST_graph(instances, root, use_prob=use_prob)

    return [(s, t, mst.get_label(s, t)) for (s, t) in  mst.iteredges()]

