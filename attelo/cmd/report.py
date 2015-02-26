"combine counts into a single report"

from __future__ import print_function
from functools import wraps
from os import path as fp
import argparse
import json
import sys

from ..args import add_common_args, add_report_args
from ..io import (load_predictions)
from ..score import (score_edges, score_edus,
                     score_edges_by_label,
                     build_confusion_matrix)
from ..report import (CombinedReport, EdgeReport, EduReport)
from ..harness.report import (ReportPack)
from .util import (load_args_data_pack,
                   get_output_dir, announce_output_dir)

# pylint: disable=too-few-public-methods

# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------


def config_argparser(psr):
    "add subcommand arguments to subparser"

    add_common_args(psr)
    add_report_args(psr)
    psr.add_argument("--predictions", metavar="FILE",
                     help="single predictions",
                     required=True)
    psr.add_argument("--fold-file", metavar="FILE",
                     type=argparse.FileType('r'),
                     help="read folds from this file")
    psr.add_argument("--fold", metavar="INT",
                     type=int,
                     help="only report on this fold")
    psr.add_argument("--output", metavar="DIR",
                     help="save report to this file")
    psr.set_defaults(func=main)


def _validate_report_args(wrapped):
    """
    Given a function that accepts an argparsed object, check
    the fold arguments before carrying on.

    The idea here is that --fold and --fold-file are meant to
    be used together (xnor)

    This is meant to be used as a decorator, eg.::

        @_validate_report_args
        def main(args):
            blah
    """
    @wraps(wrapped)
    def inner(args):
        "die if report args are incomplete"
        if args.index is not None and args.fold_file is None:
            # I'd prefer to use something like ArgumentParser.error
            # here, but I can't think of a convenient way of passing
            # the parser object in or otherwise obtaining it
            sys.exit("arg error: --fold-file is required when "
                     "--index is present")
        if args.fold is not None and args.fold_file is None:
            # I'd prefer to use something like ArgumentParser.error
            # here, but I can't think of a convenient way of passing
            # the parser object in or otherwise obtaining it
            sys.exit("arg error: --fold-file is required when "
                     "--fold is present")
        wrapped(args)
    return inner


@_validate_report_args
def main(args):
    "subcommand main (invoked from outer script)"
    output_dir = get_output_dir(args)
    dpack = load_args_data_pack(args)
    if args.fold is not None:
        fold_dict = json.load(args.fold_file)
        dpack = dpack.testing(fold_dict, args.fold)

    predictions = load_predictions(args.predictions)
    edge_counts = score_edges(dpack, predictions)
    edge_label_counts = score_edges_by_label(dpack, predictions)
    edu_counts = score_edus(dpack, predictions)
    cmatrix = build_confusion_matrix(dpack, predictions)

    rel_report = CombinedReport(EdgeReport,
                                {(k,): EdgeReport([v])
                                 for k, v in edge_label_counts})

    key = fp.basename(args.prediction)

    rpack = ReportPack(edge=CombinedReport(EdgeReport,
                                           {key: edge_counts}),
                       edu=CombinedReport(EduReport,
                                          {key: edu_counts}),
                       edge_by_rel=rel_report,
                       confusion=cmatrix,
                       confusion_labels=dpack.labels)
    rpack.dump(output_dir)
    announce_output_dir(output_dir)
