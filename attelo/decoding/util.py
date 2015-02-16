"""
Utility classes functions shared by decoders
"""

from ..edu import (FAKE_ROOT_ID)


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


def get_prob_map_raw(instances):
    """
    Reformat a probability distribution as a dictionary from
    edu pairs to a (relation, probability) tuples

    :rtype dict (EDU, EDU) (string, float)
    """
    return {(e1.id, e2.id): (rel, prob)
            for e1, e2, prob, rel in instances}


def subgroupings(sorted_edus):
    """ iterator over sorted edus -> subgrouping
    sentence by sentence

    (we exploit here the idea that subgroupings are contiguous,
    eg. as would be the case for sentences)
    """
    if not sorted_edus:
        raise ValueError('Need a non-empty list of EDUs')
    if sorted_edus[0].id == FAKE_ROOT_ID:
        sorted_edus = sorted_edus[1:]
    sentence_nb = sorted_edus[0].subgrouping
    yield sentence_nb
    for one_edu in sorted_edus[1:]:
        if one_edu.subgrouping != sentence_nb:
            sentence_nb = one_edu.subgrouping
            yield sentence_nb
