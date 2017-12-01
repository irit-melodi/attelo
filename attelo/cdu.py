"""Explicit representation of a CDU.

As of 2016-07-28, this is WIP.
"""

from collections import namedtuple


class CDU(namedtuple("CDU", "id members")):
    """A class representing the CDU (id, [members])"""
    pass


