"""
Utility classes functions shared by decoders
"""


class DecoderException(Exception):
    """
    Exceptions that arise during the decoding process
    """
    def __init__(self, msg):
        super(DecoderException, self).__init__(msg)


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
