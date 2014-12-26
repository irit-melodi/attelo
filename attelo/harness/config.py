"""
Configuring the harness
"""

# Author: Eric Kow
# License: CeCILL-B (French BSD3-like)

import argparse
from collections import namedtuple


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


# pylint: disable=pointless-string-statement, too-few-public-methods
class LearnerConfig(object):
    """
    Unique combination of learner settings.

    It is crucial that each configuration be associated with
    a unique name, as we use these names to determine if we can
    reuse a learner or not.
    """
    def __init__(self, name, attach, relate=None):
        self.name = name
        "unique name for this configuration"

        self.attach = attach
        "attachment learner"

        self.relate = relate
        "relation learner (None if same as attach)"

    @classmethod
    def simple(cls, name):
        """
        Simple learner consisting of just the attachment learner
        of the same name.
        """
        return cls(name, name)


_DecoderConfig = namedtuple("_DecoderConfig", "name decoder")


class DecoderConfig(_DecoderConfig):
    """
    Unique combination of decoder settings.
    """
    @classmethod
    def simple(cls, name):
        """
        Simple learner consisting of just the decoder of the same
        name.
        """
        return cls(name, name)
# pylint: enable=pointless-string-statement, too-few-public-methods

EvaluationConfig = namedtuple("EvaluationConfig", "name learner decoder")
