"split data into folds"

from __future__ import print_function
import argparse
import json
import sys

from ..args import (add_common_args, args_to_rng, DEFAULT_NFOLD)
from ..fold import make_n_fold
from ..io import load_data_pack


def config_argparser(psr):
    "add subcommand arguments to subparser"

    add_common_args(psr)
    psr.set_defaults(func=main)
    psr.add_argument("--nfold", "-n",
                     default=DEFAULT_NFOLD, type=int,
                     help="nfold cross-validation number "
                     "(default %d)" % DEFAULT_NFOLD)
    psr.add_argument("-s", "--shuffle",
                     default=False, action="store_true",
                     help="if set, ensure a different cross-validation "
                     "of files is done, otherwise, the same file "
                     "splitting is done everytime")
    psr.add_argument("--output", type=argparse.FileType('w'),
                     help="save folds to a json file")


def main_for_harness(args, dpack):
    """
    main function core that you can hook into if writing your own
    harness

    You have to supply the data yourself
    """
    rng = args_to_rng(args)
    fold_struct = make_n_fold(dpack, args.nfold, rng)
    json_output = args.output or sys.stdout
    json.dump(fold_struct, json_output, indent=2)


def main(args):
    "subcommand main (called from mother script)"

    dpack = load_data_pack(args.edus, args.features)
    main_for_harness(args, dpack)
    if args.output is None:
        print("")
