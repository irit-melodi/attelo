"build a discourse graph from edu pairs and a model"

from __future__ import print_function

from joblib import (Parallel)

from ..args import (add_common_args, add_decoder_args,
                    add_model_read_args,
                    add_fold_choice_args, validate_fold_choice_args,
                    args_to_decoder, args_to_decoding_mode)
from ..io import (load_model, load_fold_dict)
from ..util import Team
from .util import load_args_data_pack
import attelo.harness.decode as hdecode

# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


def _load_and_select_data(args):
    """
    read data and filter on fold if relevant
    """
    if args.fold is None:
        dpack = load_args_data_pack(args)
        return dpack
    else:
        # load fold dictionary before data pack
        # this way, if it fails we find out sooner
        # instead of waiting for the data pack
        fold_dict = load_fold_dict(args.fold_file)
        dpack = load_args_data_pack(args)
        return dpack.testing(fold_dict, args.fold)


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------


def config_argparser(psr):
    "add subcommand arguments to subparser"

    add_common_args(psr)
    add_model_read_args(psr, "model needed for {} prediction")
    psr.add_argument("--output", "-o",
                     default=None,
                     required=True,
                     metavar="FILE",
                     help="save predicted structures here")
    add_decoder_args(psr)
    add_fold_choice_args(psr)
    psr.set_defaults(func=main)


@validate_fold_choice_args
def main(args):
    "subcommand main"
    dpack = _load_and_select_data(args)
    model_paths = Team(attach=args.attachment_model,
                       relate=args.relation_model)
    models = model_paths.fmap(load_model)
    decoder = args_to_decoder(args)
    mode = args_to_decoding_mode(args)
    jobs = hdecode.jobs(dpack, models, decoder, mode, args.output)
    Parallel(n_jobs=-1, verbose=5)(jobs)
    hdecode.concatenate_outputs(dpack, args.output)
