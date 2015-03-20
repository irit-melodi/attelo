"split data into folds"

from __future__ import print_function

from ..args import (add_common_args, DEFAULT_NFOLD)
from ..fold import make_n_fold
from ..io import (save_fold_dict)
from ..util import (mk_rng)
from .util import (load_args_multipack)


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
    psr.add_argument("--output", metavar='FILE',
                     help="save folds to a json file",
                     required=True)


def main(args):
    "subcommand main (called from mother script)"
    mpack = load_args_multipack(args)
    rng = mk_rng(args.shuffle)
    fold_struct = make_n_fold(mpack.keys(), args.nfold, rng)
    save_fold_dict(fold_struct, args.output)
