"combine counts into a single report"

from __future__ import print_function
from collections import defaultdict, namedtuple
from os import path as fp
import argparse
import csv
import json
import os
import sys

import numpy
from sklearn.metrics import confusion_matrix

from ..args import add_common_args, add_report_args
from ..decoding import count_correct
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
    psr.add_argument("index_file", metavar="FILE",
                     help="json index file (see doc)")
    psr.add_argument("--fold-file", metavar="FILE",
                     type=argparse.FileType('r'),
                     required=True,
                     help="read folds from this file")
    psr.add_argument("--output", metavar="DIR",
                     help="save report to this file")
    psr.set_defaults(func=main)

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


def score_outputs(dpack, fold_dict, index):
    """read outputs mentioned in the index files and score them
    against the reference pack

    Return a single combined report
    """
    evals = defaultdict(list)
    confusion = {}
    for fold in index['folds']:
        fold_num = fold['number']
        with Torpor('scoring fold {}'.format(fold_num)):
            fpack = dpack.testing(fold_dict, fold_num)
            for config in index['configurations']:
                key = _config_key(config)
                predict_file = fp.join(fold['path'], config['predictions'])
                predictions = load_predictions(predict_file)
                # score
                evals[key].append(count_correct(fpack, predictions))
                # we store a separate confusion matrix for each config,
                # accumulating results across the folds (this should be
                # safe to do as the matrices have been forced to the
                # same shape regardless of what labels actually appear
                # in the fold)
                cmatrix = build_confusion_matrix(fpack, predictions)
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


def main_for_harness(args, dpack):
    "main for direct calls via test harness"
    output_dir = get_output_dir(args)
    fold_dict = json.load(args.fold_file)
    index = read_index(args.index_file)
    reports, confusion = score_outputs(dpack, fold_dict, index)
    if not fp.exists(output_dir):
        os.makedirs(output_dir)
    ofilename = fp.join(output_dir, 'scores.txt')
    with open(ofilename, 'w') as ostream:
        print(reports.table(), file=ostream)
    for key, matrix in confusion.items():
        ofilename = _key_filename(output_dir, 'confusion', key)
        with open(ofilename, 'w') as ostream:
            print(show_confusion_matrix(dpack.labels, matrix),
                  file=ostream)


def main(args):
    "subcommand main (invoked from outer script)"
    output_dir = get_output_dir(args)
    dpack = load_args_data_pack(args)
    main_for_harness(args, dpack)
    announce_output_dir(output_dir)
