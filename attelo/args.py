"""
Managing command line arguments
"""

from __future__ import print_function
from argparse import ArgumentTypeError
from functools import wraps
from ConfigParser import ConfigParser
import argparse
import Orange
import sys

from .decoding.astar import astar_decoder, h0, h_best, h_max, h_average
from .decoding.baseline import local_baseline, last_baseline
from .decoding.mst import mst_decoder
from .decoding.greedy import locallyGreedy
from .features import Features
from .learning.megam import MaxentLearner
from .learning.perceptron import\
    PerceptronArgs, Perceptron, StructuredPerceptron


def args_to_features(args):
    """
    Given the (parsed) command line arguments, return the set of
    core feature labels for our incoming dataset.

    If no configuration file is provided, we default to the
    Annodis experiment settings
    """
    config = ConfigParser()
    # cancels case-insensitive reading of variables.
    config.optionxform = lambda option: option
    with open(args.config) as config_file:
        config.readfp(config_file)
        metacfg = dict(config.items("Meta features"))
        return Features(source=metacfg["FirstNode"],
                        target=metacfg["SecondNode"],
                        source_span_start=metacfg["SourceSpanStart"],
                        source_span_end=metacfg["SourceSpanEnd"],
                        target_span_start=metacfg["TargetSpanStart"],
                        target_span_end=metacfg["TargetSpanEnd"],
                        grouping=metacfg["Grouping"],
                        label=metacfg["Label"])


def _mk_astar_decoder(heuristics, rfc):
    """
    Return an A* decoder using the given heuristics and
    right frontier constraint parameter
    """
    return lambda x, **kargs:\
        astar_decoder(x, heuristics=heuristics, RFC=rfc, **kargs)


def _known_heuristics():
    """
    Return a dictionary of possible A* heuristics.
    This lets us grab at the names of known heuristics
    for command line restruction
    """
    return {"average": h_average,
            "best": h_best,
            "max": h_max,
            "zero": h0}


def _known_decoders(heuristics, rfc):
    """
    Return a dictionary of possible decoders.
    This lets us grab at the names of known decoders
    """
    return {"last": last_baseline,
            "local": local_baseline,
            "locallyGreedy": locallyGreedy,
            "mst": mst_decoder,
            "astar": _mk_astar_decoder(heuristics, rfc)}


def _known_learners(decoder, features, perc_args=None):
    """
    Given the (parsed) command line arguments, return a sequence of
    learners in the order they were requested on the command line
    """

    # orange classifiers
    bayes = Orange.classification.bayes.NaiveLearner(adjust_threshold=True)
    bayes.name = "naive bayes"
    #svm = Orange.classification.svm.SVMLearnerEasy(probability = True)
    svm = Orange.classification.svm.SVMLearner(probability=True)
    svm.name = "svm"
    maxent = MaxentLearner()  # Orange.classification.logreg.LogRegLearner()
    maxent.name = "maxent"
    majority = Orange.classification.majority.MajorityLearner()
    majority.name = "majority"

    learners = {"bayes": bayes,
                "svm": svm,
                "maxent": maxent,
                "majority": majority}

    if perc_args:
        # home made perceptron
        perc = Perceptron(features,
                          nber_it=perc_args.iterations,
                          avg=perc_args.averaging)
        # home made structured perceptron
        struc_perc = StructuredPerceptron(features, decoder,
                                          nber_it=perc_args.iterations,
                                          avg=perc_args.averaging,
                                          use_prob=perc_args.use_prob)

        learners["perc"] = perc
        learners["struc_perc"] = struc_perc

    return learners


def _is_perceptron_learner_name(learner_name):
    """
    True if the given string corresponds to the command line
    argument for one of our perceptron based learners
    """
    return learner_name in ["perc", "struc_perc"]

# these are just dummy values (we just want the keys here)
KNOWN_HEURISTICS = _known_heuristics().keys()
KNOWN_DECODERS = _known_decoders([], False).keys()
KNOWN_ATTACH_LEARNERS = _known_learners(last_baseline, {},
                                        PerceptronArgs(0, False, False)).keys()
KNOWN_RELATION_LEARNERS = _known_learners(last_baseline, {}, None)


def args_to_decoder(args):
    """
    Given the (parsed) command line arguments, return the
    decoder that was requested from the command line
    """
    if args.heuristics not in _known_heuristics():
        raise ArgumentTypeError("Unknown heuristics: %s" %
                                args.heuristics)
    heuristic = _known_heuristics().get(args.heuristics, h_average)
    if not args.data_relations:
        args.rfc = "simple"

    _decoders = _known_decoders(heuristic, args.rfc)

    if args.decoder in _decoders:
        return _decoders[args.decoder]
    else:
        ArgumentTypeError("Unknown decoder: " + args.decoder)


def args_to_learners(decoder, features, args):
    """
    Given the (parsed) command line arguments, return a
    learner to use for attachment, and one to use for
    relation labeling.

    By default the relations learner is just the attachment
    learner, but the user can make a point of specifying a
    different one
    """

    perc_args = PerceptronArgs(iterations=args.nit,
                               averaging=args.averaging,
                               use_prob=args.use_prob)
    _learners = _known_learners(decoder, features, perc_args)

    if args.learner in _learners:
        attach_learner = _learners[args.learner]
    else:
        raise ArgumentTypeError("Unknown learner: " + args.learner)

    has_perc = _is_perceptron_learner_name(args.learner)
    if has_perc and not args.relation_learner:
        msg = "The learner '" + args.learner + "' needs a" +\
            "a non-perceptron relation learner to go with it"
        raise ArgumentTypeError(msg)
    if not args.relation_learner:
        relation_learner = attach_learner
    elif args.relation_learner in _learners:
        relation_learner = _learners[args.relation_learner]
    else:
        raise ArgumentTypeError("Unknown relation learner: "
                                + args.relation_learner)

    return attach_learner, relation_learner


def args_to_threshold(model, decoder, requested=None, default=0.5):
    """Given a model and decoder, return a threshold if

    * we request a specific threshold
    * or the decoder absolutely requires one

    In these cases, we try to return one of the following thresholds
    in order:

    1. that supplied by the model (if there is one)
    2. the requested threshold (if supplied)
    3. a default value
    """
    if requested or str(decoder.__name__) == "local_baseline":
        try:
            threshold = model.threshold
        except:
            threshold = requested if requested else default
            print("threshold forced at : ", threshold, file=sys.stderr)
    else:
        threshold = None
    return threshold

# ---------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------


def add_common_args(psr):
    "add usual attelo args to subcommand parser"

    psr.add_argument("data_attach", metavar="FILE",
                     help="attachment data")
    psr.add_argument("data_relations", metavar="FILE", nargs="?",
                     help="relations data")  # optional
    psr.add_argument("--config", "-C", metavar="FILE",
                     required=True,
                     default=None,
                     help="corpus specificities config file; if "
                     "absent, defaults to hard-wired annodis config")
    psr.add_argument("--quiet", action="store_true",
                     help="Supress all feedback")


def add_common_args_lite(psr):
    "variant of add_common_args without relations table"

    psr.add_argument("data_attach", metavar="FILE",
                     help="attachment data")
    psr.add_argument("--config", "-C", metavar="FILE",
                     required=True,
                     default=None,
                     help="corpus specificities config file; if "
                     "absent, defaults to hard-wired annodis config")


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
        if args.fold_file and not args.fold:
            # I'd prefer to use something like ArgumentParser.error
            # here, but I can't think of a convenient way of passing
            # the parser object in or otherwise obtaining it
            sys.exit("arg error: --fold is required when "
                     "--fold-file is present")
        elif args.fold and not args.fold_file:
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
    decoder_grp.add_argument("--decoder", "-d", default="local",
                             choices=KNOWN_DECODERS,
                             help="decoders for attachment "
                             "(cf also heuristics for astar)")

    astar_grp = psr.add_argument_group("A* decoder arguments")
    astar_grp.add_argument("--heuristics", "-e",
                           default="average",
                           choices=KNOWN_HEURISTICS,
                           help="heuristics used for astar decoding; "
                           "default=average")
    astar_grp.add_argument("--rfc", "-r",
                           default="full",
                           choices=["full", "simple", "none"],
                           help="with astar decoding, what kind of RFC is "
                           "applied: simple of full; simple means "
                           "everything is subordinating")

    perc_grp = psr.add_argument_group('perceptron arguments')
    perc_grp.add_argument("--use_prob", "-P",
                          default=True, action="store_false",
                          help="convert perceptron scores "
                          "into probabilities")

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
                     choices=KNOWN_ATTACH_LEARNERS,
                     help="learner for attachment [and relations]")
    psr.add_argument("--relation-learner",
                     choices=KNOWN_RELATION_LEARNERS,
                     help="learners for relation labeling "
                     "[default same as attachment]")

    # perceptron prefs
    perc_grp = groups.get("perceptron",
                          psr.add_argument_group('perceptron arguments'))
    perc_grp.add_argument("--averaging", "-m",
                          default=False, action="store_true",
                          help="averaged perceptron")
    perc_grp.add_argument("--nit", "-i",
                          default=1, type=int,
                          help="number of iterations for "
                          "perceptron models")
