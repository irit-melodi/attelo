"""
Utility classes functions shared by decoders
"""

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
