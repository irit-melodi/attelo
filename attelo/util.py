'''
General-purpose classes and functions
'''

from argparse import ArgumentTypeError
from collections import namedtuple
import enum
import itertools
import random
# pylint: disable=too-few-public-methods

RNG_SEED = "just an illusion"


class ArgparserEnum(enum.Enum):
    '''
    An enumeration whose values we spit out as choices to argparser
    '''
    @classmethod
    def choices_str(cls):
        "available choices in this enumeration"
        return ",".join(sorted(x.name for x in cls))

    @classmethod
    def help_suffix(cls, default):
        "help text suffix showing choices and default"
        if default is None:
            template = "(choices: {{{choices}}})"
            return template.format(choices=cls.choices_str())
        else:
            template = "(choices: {{{choices}}}, default: {default})"
            return template.format(choices=cls.choices_str(),
                                   default=default.name)

    @classmethod
    def from_string(cls, string):
        "from command line arg"
        names = {x.name: x for x in cls}
        value = names.get(string)
        if value is not None:
            return value
        else:
            oops = "invalid choice: {}, choose from {}"
            raise ArgumentTypeError(oops.format(string, cls.choices_str()))


class Team(namedtuple("Team", "attach label")):
    """
    Any collection where we have the same thing but duplicated
    for each attelo subtask (eg. models, learners,)
    """
    def fmap(self, func):
        """
        Apply a function to each member of the collection
        """
        return Team(attach=func(self.attach),
                    label=func(self.label))


def truncate(text, width):
    """
    Truncate a string and append an ellipsis if truncated
    """
    return text if len(text) < width else text[:width] + '...'


def concat_i(iters):
    """
    Merge an iterable of iterables into a single iterable
    """
    return itertools.chain.from_iterable(iters)


def concat_l(iters):
    """
    Merge an iterable of iterables into a list
    """
    return list(concat_i(iters))


def mk_rng(shuffle=False, default_seed=None):
    """
    Return a random number generator instance, hard-seeded
    unless we ask for shuffling to be enabled

    (note: if shuffle mode is enable, the rng in question
    will just be the system generator)
    """
    if shuffle:
        return random
    else:
        rng = random.Random()
        rng.seed(default_seed or RNG_SEED)
        return rng
