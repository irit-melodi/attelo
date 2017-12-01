"""
Managing command line arguments
"""

from __future__ import print_function
from functools import wraps
import sys

# pylint: disable=too-few-public-methods

# ---------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------


def add_common_args(psr):
    "add usual attelo args to subcommand parser"

    psr.add_argument("edus", metavar="FILE",
                     help="EDU input file (tab separated)")
    psr.add_argument("pairings", metavar="FILE",
                     help="EDU pairings file (tab separated)")
    psr.add_argument("features", metavar="FILE",
                     help="EDU pair features (libsvm)")
    psr.add_argument("vocab", metavar="FILE",
                     help="feature vocabulary")
    psr.add_argument("labels", metavar="FILE",
                     help="labels")
    psr.add_argument("--quiet", action="store_true",
                     help="Supress all feedback")


def add_fold_choice_args(psr):
    "ability to select a subset of the data according to a fold"

    fold_grp = psr.add_argument_group('fold selection')
    fold_grp.add_argument("--fold-file", metavar="FILE",
                          help="read folds from this file")
    fold_grp.add_argument("--fold", metavar="INT",
                          type=int,
                          help="fold to select")


def validate_fold_choice_args(wrapped):
    """
    Given a function that accepts an argparsed object, check
    the fold arguments before carrying on.

    The idea here is that --fold and --fold-file are meant to
    be used together (xnor)

    This is meant to be used as a decorator, eg.::

        @validate_fold_choice_args
        def main(args):
            blah
    """
    @wraps(wrapped)
    def inner(args):
        "die if fold args are incomplete"
        if args.fold_file is not None and args.fold is None:
            # I'd prefer to use something like ArgumentParser.error
            # here, but I can't think of a convenient way of passing
            # the parser object in or otherwise obtaining it
            sys.exit("arg error: --fold is required when "
                     "--fold-file is present")
        elif args.fold is not None and args.fold_file is None:
            sys.exit("arg error: --fold-file is required when "
                     "--fold is present")
        wrapped(args)
    return inner


def add_model_read_args(psr, help_):
    """
    models files we can read in

    :param help_: python format string for help `{}` will
                  have a word (eg. 'attachment') plugged in
    :type help_: string
    """

    grp = psr.add_argument_group('models')
    grp.add_argument("--attachment-model", "-A",
                     default=None,
                     required=True,
                     help=help_.format("attachment"))
    grp.add_argument("--relation-model", "-R",
                     default=None,
                     required=True,
                     help=help_.format("relations"))


def add_report_args(psr):
    """
    add args to scoring/evaluation
    """
    score_grp = psr.add_argument_group('scoring arguments')
    score_grp.add_argument("--correction", "-c",
                           default=1.0, type=float,
                           help="if input is already a restriction on the "
                           "full task, this options defines a correction to "
                           "apply on the final recall score to have the real "
                           "scores on the full corpus")
    score_grp.add_argument("--accuracy", "-a",
                           default=False, action="store_true",
                           help="provide accuracy scores for classifiers used")
