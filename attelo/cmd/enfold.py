"split data into folds"

from __future__ import print_function
import argparse
import json
import random
import sys

from ..args import\
    add_common_args_lite,\
    args_to_phrasebook, DEFAULT_NFOLD
from ..fold import make_n_fold
from ..io import read_data


def _prepare_folds(phrasebook, num_folds, table, shuffle=True):
    """Return an N-fold validation setup respecting a property where
    examples in the same grouping stay in the same fold.
    """
    if shuffle:
        random.seed()
    else:
        random.seed("just an illusion")

    return make_n_fold(table,
                       folds=num_folds,
                       meta_index=phrasebook.grouping)


def config_argparser(psr):
    "add subcommand arguments to subparser"

    add_common_args_lite(psr)
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


def main_for_harness(args, data_attach):
    """
    main function core that you can hook into if writing your own
    harness

    You have to supply the data yourself
    """
    phrasebook = args_to_phrasebook(args)
    fold_struct = _prepare_folds(phrasebook,
                                 args.nfold,
                                 data_attach,
                                 shuffle=args.shuffle)

    json_output = args.output or sys.stdout
    json.dump(fold_struct, json_output, indent=2)


def main(args):
    "subcommand main (called from mother script)"

    data_attach, _ = read_data(args.data_attach, None, verbose=True)
    main_for_harness(args, data_attach)
    if args.output is None:
        print("")
