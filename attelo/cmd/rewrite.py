"save input tables in other handy formats"

from __future__ import print_function
from os import path as fp

from .util import (get_output_dir, announce_output_dir,
                   load_args_multipack)
from ..args import add_common_args
from ..fold import (select_testing)
from ..io import (load_fold_dict, write_predictions_output)


def config_argparser(psr):
    "add subcommand arguments to subparser"

    add_common_args(psr)
    psr.add_argument("fold_file", metavar="FILE",
                     help="read folds from this file")
    psr.add_argument("--output", metavar="DIR",
                     help="output directory for graphs")
    psr.set_defaults(func=main)


def gold_predictions(dpack):
    """return the gold standard predictions as a decoder might
    (returns a single prediction only, not a list)
    """
    pairs = zip(dpack.pairings, dpack.target)
    return [(e1.id, e2.id, dpack.get_label(t))
            for (e1, e2), t in pairs]


def main_for_harness(args):
    """
    main function core that you can hook into if writing your own
    harness

    You have to supply (and filter) the data yourself
    (see `select_data`)
    """
    output_dir = get_output_dir(args)
    mpack = load_args_multipack(args)
    fold_dict = load_fold_dict(args.fold_file)
    for fold in set(fold_dict.values()):
        fpack = select_testing(mpack, fold_dict, fold)
        filename = fp.join(output_dir, "gold-" + str(fold))
        write_predictions_output(fpack, gold_predictions(fpack), filename)
    announce_output_dir(output_dir)


def main(args):
    "subcommand main (invoked from outer script)"

    main_for_harness(args)
