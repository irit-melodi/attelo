"""
Utility classes functions shared by decoders
"""

import numpy as np

from attelo.table import (Graph, UNRELATED)


class DecoderException(Exception):
    """
    Exceptions that arise during the decoding process
    """
    pass


def get_sorted_edus(instances):
    """
    Return a list of EDUs, using the following as sort key in order of

    * starting position (earliest edu first)
    * ending position (narrowest edu first)

    Note that there may be EDU pairs with the same spans
    (particularly in case of annotation error). In case of ties,
    the order should be considered arbitrary
    """

    edus = set()
    for edu1, edu2, _, _ in instances:
        edus.add(edu1)
        edus.add(edu2)

    return sorted(edus, key=lambda x: x.span())


def get_prob_map(instances):
    """
    Reformat a probability distribution as a dictionary from
    edu id pairs to a (relation, probability) tuples

    :rtype dict (string, string) (string, float)
    """
    return {(e1.id, e2.id): (rel, prob)
            for e1, e2, prob, rel in instances}


def convert_prediction(dpack, triples):
    """Populate a datapack prediction array from a list
    of triples

    Parameters
    ----------
    prediction: [(string, string, string)]

        List of EDU id, EDU id, label triples

    Returns
    -------
    dpack: DataPack
        A copy of the original DataPack with predictions
        set
    """
    link_map = {(id1, id2): lab for id1, id2, lab in triples}

    def get_lbl(pair):
        'from edu pair to label number'
        edu1, edu2 = pair
        key = edu1.id, edu2.id
        lbl = link_map.get(key, UNRELATED)
        return dpack.label_number(lbl)
    prediction = np.fromiter((get_lbl(pair) for pair in dpack.pairings),
                             dtype=np.dtype(np.int16))
    graph = Graph(prediction=prediction,
                  attach=dpack.graph.attach,
                  label=dpack.graph.label)
    return dpack.set_graph(graph)


def simple_candidates(dpack):
    '''
    Translate the links into a list of (EDU, EDU, float, string)
    quadruplets representing the attachment probability and the
    the best label for each EDU pair.  This is often good enough
    for simplistic decoders
    '''
    if dpack.graph is None:
        raise ValueError("Tried to extract weights from an "
                         "unweighted datapack")
    wts = dpack.graph
    best_lbls = np.ravel(np.argmax(wts.label, axis=1))
    return [(pair[0], pair[1], score, dpack.get_label(lbl))
            for pair, score, lbl
            in zip(dpack.pairings, wts.attach, best_lbls)]


def prediction_to_triples(dpack):
    """
    Returns
    -------
    triples: prediction: [(string, string, string)]

        List of EDU id, EDU id, label triples
        omitting the unrelated triples
    """
    if dpack.graph is None:
        raise ValueError("Not a weighted datapack")
    unrelated = dpack.label_number(UNRELATED)
    return [(edu1.id, edu2.id, dpack.get_label(lbl))
            for (edu1, edu2), lbl in
            zip(dpack.pairings, dpack.graph.prediction)
            if lbl != unrelated]
