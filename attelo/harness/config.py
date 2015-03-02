"""
Configuring the harness
"""

# Author: Eric Kow
# License: CeCILL-B (French BSD3-like)

import argparse
from collections import namedtuple

from attelo.util import Team

# pylint: disable=too-few-public-methods


# pylint: disable=abstract-class-not-used
class CliArgs(argparse.Namespace):
    """
    An abstract context manager that you would need to extend.
    This allows you to define the :py:class:`argparse.Namespace`
    objects that attelo creates when parsing its command line
    arguments. It's useful if you want to bypass the command line
    and call parts of attelo commands by hand

    You would tend to use it like so ::

        with YourCliManager as args:
            ...
            attelo.cmd.foo.main_for_harness(args, other_args)
            ...

    This is defined as a context manager partly so that you have
    the opportunity to close any file handles that get opened on
    CLI parsing time (which may be important if you're doing
    something equivalent to calling a program many many times)

    Note that when implementing the `__exit__` method,
    the namespace object is `self` (so you'd write eg
    `self.output_file.close()`)
    """

    def parser(self):
        """
        The argparser that would be called on context manager
        entry

        Your subclass should probably generate an
        :py:class:`argparse.ArgumentParser()` and instantiate it
        with some commands' `config_argparser` methods
        """
        raise NotImplementedError()

    def argv(self):
        """
        Command line arguments that would correspond to this
        configuration

        :rtype: `[String]`
        """
        raise NotImplementedError()

    def __enter__(self):
        psr = self.parser()
        argv = self.argv()
        psr.parse_args(argv, self)
        return self

    def __exit__(self, ctype, value, traceback):
        """
        Tidy up any open file handles, etc

        This can be useful if you are calling commands that tend
        to open files passed in from the command line via the
        :py:class:`argparse.FileType` method
        """
        return


class Variant(namedtuple("Variant", "key name flags")):
    """
    Unique combination of settings for either a learner or
    decoder

    :param name: name of the learner/decoder
    :type name: string

    :param key: unique string that distinguishes this learner/decoder
                from others (eg. 'astar-2-best')
    :type key: string

    :param flags: command line flags
    :type flags: [string]
    """
    @classmethod
    def simple(cls, name):
        """
        Simple variant where the key is just the name of component
        itself
        """
        return cls(name, name, [])


class Keyed(namedtuple('Keyed', 'key payload')):
    '''
    A keyed object is just any object that is attached with a
    short unique (mnemonic) identifier.

    Keys often appear in filenames so it's best to avoid
    whitespace, fancy characters, and for portability reasons,
    anything non-ASCII.
    '''
    pass


class Learner(namedtuple("Learner", "key name flags decoder")):
    """
    Unique combination of settings for a learner.
    See :py:class:`Variant`

    :param decoder: decoder that this learner is parameterised by
    :type decoder: Variant or None
    """
    pass


class LearnerConfig(Team):
    """
    Combination of an attachment and a relation learner variant

    :type attach: Learner
    :type relate: Learner
    """
    def __new__(cls, attach, relate):
        team = super(LearnerConfig, cls).__new__(cls, attach, relate)
        team.key = team.attach.key
        if team.relate is not None:
            team.key += "_" + team.relate.key
        return team


class EvaluationConfig(namedtuple("EvaluationConfig",
                                  "key settings learner decoder")):
    """
    Combination of learners, decoders and decoder settings
    for an attelo evaluation

    The settings can really be of type that has a 'key'
    field; but you should have a way of extracting at
    least a :pyclass:`DecodingMode` from it

    :type learner: Keyed (Team learner)
    :type decoder: Keyed Decoder
    :type settings: Keyed (???)
    """
    @classmethod
    def simple_key(cls, learner, decoder):
        """
        generate a short unique name for a learner/decoder combo
        """
        return "%s-%s" % (learner.key, decoder.key)
