"combine counts into a single report"

from __future__ import print_function
from collections import defaultdict, namedtuple
import argparse
import csv
import json
import sys

from ..args import add_report_args
from ..report import Count, Report, CombinedReport


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

    add_report_args(psr)
    psr.add_argument("index_file", metavar="FILE",
                     type=argparse.FileType('r'),
                     help="csv index file (see doc)")
    psr.add_argument("--json", metavar="FILE",
                     type=argparse.FileType('w'),
                     help="save detailed reports to json file")
    psr.set_defaults(func=main)

# TODO: we need to revist the question of params in the reporting code
NullParams = namedtuple("NullParams", "dummy")


def main(args):
    "subcommand main (invoked from outer script)"
    evals = defaultdict(list)
    fold_evals = defaultdict(lambda: defaultdict(list))
    for row in _read_index_file(args.index_file):
        with open(row["counts_file"]) as cfile:
            config = row["config"]
            fold = row["fold"]
            counts = Count.read_csv(cfile)
            evals[config].extend(counts.values())
            fold_evals[config][fold].extend(counts.values())
    reports = CombinedReport({k: Report(v, params=NullParams(dummy=None))
                              for k, v in evals.items()})
    print(reports.table())
    if args.json is not None:
        json.dump(reports.for_json(), args.json, indent=2)
