'''
General-purpose classes and functions
'''

from collections import namedtuple
# pylint: disable=too-few-public-methods


class Team(namedtuple("Team", "attach relate")):
    """
    Any collection where we have the same thing but duplicated
    for each attelo subtask (eg. models, learners,)
    """
    pass


def truncate(text, width):
    """
    Truncate a string and append an ellipsis if truncated
    """
    return text if len(text) < width else text[:width] + '...'
