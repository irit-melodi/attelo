"""
Managing command line arguments
"""

from __future__ import print_function
from argparse import ArgumentTypeError
from functools import wraps
import argparse
import random
import sys

# pylint: disable=no-name-in-module
# pylint at the time of this writing doesn't deal well with
# packages that dynamically generate methods
# https://bitbucket.org/logilab/pylint/issue/58/false-positive-no-member-on-numpy-imports
from numpy import inf
# pylint: enable-no-name-in-module

from .decoding import (DecodingMode, DecoderArgs, DECODERS)
from .decoding.astar import (AstarArgs, RfcConstraint, Heuristic)
from .decoding.mst import (MstRootStrategy)
from .learning import (LearnerArgs, PerceptronArgs,
                       ATTACH_LEARNERS, RELATE_LEARNERS)
from .util import Team
# pylint: disable=too-few-public-methods


def _is_perceptron_learner_name(learner_name):
    """
    True if the given string corresponds to the command line
    argument for one of our perceptron based learners
    """
    return learner_name in ["perc", "struc_perc"]

# default values for perceptron learner
DEFAULT_PERCEPTRON_ARGS = PerceptronArgs(iterations=20,
                                         averaging=True,
                                         use_prob=True,
                                         aggressiveness=inf)

DEFAULT_MST_ROOT = MstRootStrategy.fake_root

# default values for A* decoder
# (NB: not the same as in the default initialiser)
DEFAULT_ASTAR_ARGS = AstarArgs(rfc=RfcConstraint.full,
                               heuristics=Heuristic.average,
                               beam=None,
                               nbest=1)
DEFAULT_HEURISTIC = DEFAULT_ASTAR_ARGS.heuristics
DEFAULT_BEAMSIZE = DEFAULT_ASTAR_ARGS.beam
DEFAULT_NBEST = DEFAULT_ASTAR_ARGS.nbest
DEFAULT_RFC = DEFAULT_ASTAR_ARGS.rfc

#
DEFAULT_DECODER = "local"
DEFAULT_NIT = DEFAULT_PERCEPTRON_ARGS.iterations
DEFAULT_NFOLD = 10

# these are just dummy values (we just want the keys here)
KNOWN_DECODERS = DECODERS.keys()

RNG_SEED = "just an illusion"


def args_to_rng(args):
    """
    Return a random number generator instance, hard-seeded
    unless we ask for shuffling to be enabled

    (note: if shuffle mode is enable, the rng in question
    will just be the system generator)
    """
    if args.shuffle:
        return random
    else:
        rng = random.Random()
        rng.seed(RNG_SEED)
        return rng


def args_to_decoder(args):
    """
    Given the parsed command line arguments, and an attachment model, return
    the decoder that was requested from the command line
    """
    args.rfc = RfcConstraint.simple

    astar_args = AstarArgs(rfc=args.rfc,
                           heuristics=args.heuristics,
                           beam=args.beamsize,
                           nbest=args.nbest)

    config = DecoderArgs(threshold=args.threshold,
                         mst_root_strategy=args.mst_root_strategy,
                         astar=astar_args,
                         use_prob=not args.non_prob_scores)

    if args.decoder in DECODERS:
        factory = DECODERS[args.decoder]
        return factory(config)
    else:
        raise ArgumentTypeError("Unknown decoder: " + args.decoder)


def args_to_decoding_mode(args):
    """
    Return configuration tuple for the decoding operation
    """
    if args.post_label:
        return DecodingMode.post_label
    else:
        return DecodingMode.joint


def _get_learner(name, is_for_attach=True):
    '''
    Return learner constructor that goes with the given
    name or raise an ArgumentTypeError
    '''

    if is_for_attach:
        ldict = ATTACH_LEARNERS
        desc = 'attachment'
    else:
        ldict = RELATE_LEARNERS
        desc = 'relation labelling'

    if not (name in ATTACH_LEARNERS or name in RELATE_LEARNERS):
        # completely unknown
        raise ArgumentTypeError("Unknown learner: " + name)
    elif name not in ldict:
        raise ArgumentTypeError(('The learner "{}" cannot be used for the '
                                 '{} task').format(name, desc))
    else:
        return ldict[name]


def _get_learner_set(args):
    '''
    Return a pair of learner wrappers (not the actual
    learners, which we would need the other args for)

    :rtype Team(learner wrapper)
    '''
    aname = args.learner
    rname = args.learner if args.relation_learner is None\
        else args.relation_learner

    has_perc = _is_perceptron_learner_name(aname)
    if has_perc and not args.relation_learner:
        msg = "The learner '" + args.learner + "' needs a" +\
            "a non-perceptron relation learner to go with it"
        raise ArgumentTypeError(msg)

    attach_learner = _get_learner(aname, is_for_attach=True)
    relate_learner = _get_learner(rname, is_for_attach=False)

    return Team(attach=attach_learner,
                relate=relate_learner)


def args_to_learners(decoder, args):
    """
    Given the (parsed) command line arguments, return a
    learner to use for attachment, and one to use for
    relation labeling.

    By default the relations learner is just the attachment
    learner, but the user can make a point of specifying a
    different one

    :rtype Team(learner)
    """

    perc_args = PerceptronArgs(iterations=args.nit,
                               averaging=args.averaging,
                               use_prob=not args.non_prob_scores,
                               aggressiveness=args.aggressiveness)
    learner_args = LearnerArgs(decoder=decoder,
                               perc_args=perc_args)

    wrappers = _get_learner_set(args)
    return Team(attach=wrappers.attach(learner_args),
                relate=wrappers.relate(learner_args))

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
    psr.add_argument("--config", "-C", metavar="FILE",
                     required=True,
                     default=None,
                     help="corpus specificities config file; if "
                     "absent, defaults to hard-wired annodis config")
    psr.add_argument("--quiet", action="store_true",
                     help="Supress all feedback")


def add_fold_choice_args(psr):
    "ability to select a subset of the data according to a fold"

    fold_grp = psr.add_argument_group('fold selection')
    fold_grp.add_argument("--fold-file", metavar="FILE",
                          type=argparse.FileType('r'),
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


def add_decoder_args(psr):
    """
    add decoding related args to subcommand parser

    NB: already included by add_learner_args
    """
    _add_decoder_args(psr)


def _add_decoder_args(psr):
    """
    core of add_decoder_args, returns a dictionary of groups
    """

    decoder_grp = psr.add_argument_group('decoder arguments')
    decoder_grp.add_argument("--threshold", "-t",
                             default=None, type=float,
                             help="force the classifier to use this threshold "
                             "value for attachment decisions, unless it is "
                             "trained explicitely with a threshold")
    decoder_grp.add_argument("--decoder", "-d", default=DEFAULT_DECODER,
                             choices=KNOWN_DECODERS,
                             help="decoders for attachment; " +
                             "default: %s " % DEFAULT_DECODER +
                             "(cf also heuristics for astar)")

    mst_grp = psr.add_argument_group("MST/MSDAG decoder arguments")
    mst_grp.add_argument("--mst-root-strategy",
                         default=DEFAULT_MST_ROOT,
                         type=MstRootStrategy.from_string,
                         help="how the MST decoder should select its root "
                         + MstRootStrategy.help_suffix(DEFAULT_MST_ROOT))

    astar_grp = psr.add_argument_group("A* decoder arguments")
    astar_grp.add_argument("--heuristics", "-e",
                           default=DEFAULT_HEURISTIC,
                           type=Heuristic.from_string,
                           help="heuristics used for astar decoding; "
                           "default: %s" % DEFAULT_HEURISTIC.name)
    astar_grp.add_argument("--rfc", "-r",
                           type=RfcConstraint.from_string,
                           default=DEFAULT_RFC,
                           help="with astar decoding, what kind of RFC is "
                           "applied: simple of full; simple means everything "
                           "is subordinating (default: %s)" % DEFAULT_RFC.name)
    astar_grp.add_argument("--beamsize", "-B",
                           default=DEFAULT_BEAMSIZE,
                           type=int,
                           help="with astar decoding, set a beamsize "
                           "default: None -> full astar with")

    astar_grp.add_argument("--nbest", "-N",
                           default=DEFAULT_NBEST,
                           type=int,
                           help="with astar decoding, set a nbest oracle, "
                           "keeping n solutions "
                           "default: 1-best = simple astar")

    perc_grp = psr.add_argument_group('perceptron arguments')
    perc_grp.add_argument("--non-prob-scores",
                          default=not DEFAULT_PERCEPTRON_ARGS.use_prob,
                          action="store_true",
                          help="do NOT treat scores as probabilities")

    # harness prefs (shared between eval)
    psr.add_argument("--post-label", "-p",
                     default=False, action="store_true",
                     help="decode only on attachment, and predict "
                     "relations afterwards")

    return {"astar": astar_grp,
            "perceptron": perc_grp,
            "decoder": decoder_grp}


def add_learner_args(psr):
    """
    add classifier related args to subcommand parser

    NB: includes add_decoder_args
    """

    groups = _add_decoder_args(psr)

    psr.add_argument("--learner", "-l",
                     default="bayes",
                     choices=ATTACH_LEARNERS.keys(),
                     help="learner for attachment [and relations]")
    psr.add_argument("--relation-learner",
                     choices=RELATE_LEARNERS.keys(),
                     help="learners for relation labeling "
                     "[default same as attachment]")

    # perceptron prefs
    perc_grp = groups.get("perceptron",
                          psr.add_argument_group('perceptron arguments'))
    perc_grp.add_argument("--averaging", "-m",
                          default=DEFAULT_PERCEPTRON_ARGS.averaging,
                          action="store_true",
                          help="averaged perceptron")
    perc_grp.add_argument("--nit", "-i",
                          default=DEFAULT_PERCEPTRON_ARGS.iterations,
                          type=int,
                          help="number of iterations for "
                          "perceptron models (default: %d)" %
                          DEFAULT_PERCEPTRON_ARGS.iterations)
    perc_grp.add_argument("--aggressiveness",
                          default=DEFAULT_PERCEPTRON_ARGS.aggressiveness,
                          type=float,
                          help="aggressivness (passive-aggressive perceptrons "
                          "only); (default: %f)" %
                          DEFAULT_PERCEPTRON_ARGS.aggressiveness)


def validate_learner_args(wrapped):
    """
    Given a function that accepts an argparsed object, check
    the learner arguments before carrying on.

    This is meant to be used as a decorator, eg.::

        @validate_learner_args
        def main(args):
            blah
    """
    @wraps(wrapped)
    def inner(args):
        "die if learner are invalid"
        try:
            _get_learner_set(args)
        except ArgumentTypeError as err:
            sys.exit('arg error: ' + str(err))
        wrapped(args)
    return inner


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
