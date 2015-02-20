"combine counts into a single report"

from __future__ import print_function
from collections import defaultdict, namedtuple
from functools import wraps
from os import path as fp
import argparse
import csv
import json
import os
import sys

import numpy
from sklearn.metrics import confusion_matrix

from ..args import add_common_args, add_report_args
from ..decoding import (count_correct_edges, count_correct_edus)
from ..io import load_predictions, Torpor
from ..report import (CombinedReport, Report,
                      show_confusion_matrix)
from .util import (load_args_data_pack,
                   get_output_dir, announce_output_dir)


EXPECTED_KEYS = ["config", "fold", "counts_file"]

# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


def _read_index_file(fstream):
    "read the index file into a simple dict"
    reader = csv.DictReader(fstream, fieldnames=EXPECTED_KEYS)
    header_row = reader.next()
    header = [header_row[k] for k in EXPECTED_KEYS]
    if header != EXPECTED_KEYS:
        sys.exit("Malformed index file (expected keys: %s, got: %s)"
                 % (EXPECTED_KEYS, header))
    return list(reader)


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------


def config_argparser(psr):
    "add subcommand arguments to subparser"

    add_common_args(psr)
    add_report_args(psr)
    input_grp = psr.add_mutually_exclusive_group(required=True)
    input_grp.add_argument("--index", metavar="FILE",
                           help="json index file (see doc)")
    input_grp.add_argument("--predictions", metavar="FILE",
                           help="single predictions")
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


# TODO: we need to revist the question of params in the reporting code
NullParams = namedtuple("NullParams", "dummy")


def _make_relative_wrt(index_file, path):
    """return modified path, treating relative paths as relative
    to the directory that the index file is in

    if the parent dir is absolute, this resulting path will be
    absolute; otherwise it will merely be relative in the same
    way that the parent dir is
    """
    dname = fp.dirname(index_file)
    return path if fp.isabs(path) else fp.join(dname, path)


def _config_key(item):
    """return an attelo report table key"""
    learner = item['attach-learner']
    if 'relate-learner' in item:
        learner += ':' + item['relate-learner']
    return (learner, item['decoder'])


def read_index(master_index):
    """read master index and nested fold index files;
    and return result as a single combined dictionary
    """
    with open(master_index, 'r') as index_stream:
        index = json.load(index_stream)
        for fold in index['folds']:
            fold['path'] = _make_relative_wrt(master_index, fold['path'])
    return index


def fake_index(predictions_file, fold):
    """return a fake index dictionary for a
    single prediction file
    """
    return {'folds': [{'number': fold,
                       'path': fp.dirname(predictions_file)}],
            'configurations': [{'attach-learner': 'x',
                                'decoder': 'x',
                                'predictions': fp.basename(predictions_file)}]}


def build_confusion_matrix(dpack, predictions):
    """return a confusion matrix show predictions vs desired labels
    """
    pred_target = [dpack.label_number(label) for _, _, label in predictions]
    # we want the confusion matrices to have the same shape regardless
    # of what labels happen to be used in the particular fold
    # pylint: disable=no-member
    labels = numpy.arange(1, len(dpack.labels) + 1)
    # pylint: enable=no-member
    return confusion_matrix(dpack.target, pred_target, labels)


def score_predictions(dpack, predict_file):
    """score the given predictions against the data pack,
    returning counts and a confusion matrix
    """
    predictions = load_predictions(predict_file)
    # score
    evals = count_correct_edges(dpack, predictions)
    edu_counts = count_correct_edus(dpack, predictions)
    cmatrix = build_confusion_matrix(dpack, predictions)
    return evals, edu_counts, cmatrix


def _prediction_file(fold, config):
    "predictions file for a given config within a fold"
    return fp.join(fold['path'], config['predictions'])


def _score_fold(dpack, fold_dict, index, fold):
    "scores for all configs within a fold"
    fold_num = fold['number']
    fpack = dpack.testing(fold_dict, fold_num)
    return [score_predictions(fpack, _prediction_file(fold, c))
            for c in index['configurations']]


def score_outputs(dpack, fold_dict, index):
    """read outputs mentioned in the index files and score them
    against the reference pack

    Return a single combined report
    """
    evals = defaultdict(list)
    confusion = {}
    configs = index['configurations']
    for fold in index['folds']:
        fold_num = fold['number']
        with Torpor('scoring fold {}'.format(fold_num)):
            scores = _score_fold(dpack, fold_dict, index, fold)
            for config, (counts, ecounts, cmatrix) in zip(configs, scores):
                key = _config_key(config)
                # score
                evals[key].append((counts, ecounts))
                # we store a separate confusion matrix for each config,
                # accumulating results across the folds (this should be
                # safe to do as the matrices have been forced to the
                # same shape regardless of what labels actually appear
                # in the fold)
                if key in confusion:
                    confusion[key] += cmatrix
                else:
                    confusion[key] = cmatrix
    reports = CombinedReport({k: Report(v, params=NullParams(dummy=None))
                              for k, v in evals.items()})
    return reports, confusion


def _key_filename(output_dir, prefix, key):
    'from config key to filename'
    bname = '-'.join([prefix] + list(key))
    return fp.join(output_dir, bname)


def main_for_harness(args, dpack, output_dir):
    "main for direct calls via test harness"
    if args.index is not None:
        index = read_index(args.index)
        fold_dict = json.load(args.fold_file)
    elif args.fold is not None:
        index = fake_index(args.predictions, args.fold)
        fold_dict = json.load(args.fold_file)
    else:
        index = fake_index(args.predictions, 0)
        fold_dict = {k: 0 for k in dpack.groupings()}
    reports, confusion = score_outputs(dpack, fold_dict, index)
    if not fp.exists(output_dir):
        os.makedirs(output_dir)
    # edgewise scores
    ofilename = fp.join(output_dir, 'scores.txt')
    with open(ofilename, 'w') as ostream:
        print(reports.edge_table(), file=ostream)
    # edu scores
    ofilename = fp.join(output_dir, 'edu-scores.txt')
    with open(ofilename, 'w') as ostream:
        print(reports.edu_table(), file=ostream)
    # confusion matrices
    for key, matrix in confusion.items():
        ofilename = _key_filename(output_dir, 'confusion', key)
        with open(ofilename, 'w') as ostream:
            print(show_confusion_matrix(dpack.labels, matrix),
                  file=ostream)


@_validate_report_args
def main(args):
    "subcommand main (invoked from outer script)"
    output_dir = get_output_dir(args)
    dpack = load_args_data_pack(args)
    main_for_harness(args, dpack, output_dir)
    announce_output_dir(output_dir)
