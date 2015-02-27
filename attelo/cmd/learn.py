"learn and save models"

from __future__ import print_function

from joblib import (Parallel)

import attelo.harness.learn as hlearn
from ..args import\
    (add_common_args,
     add_learner_args, validate_learner_args,
     add_fold_choice_args, validate_fold_choice_args,
     args_to_decoder, args_to_learners)
from ..io import (load_fold_dict)
from ..learning import (Task)
from ..table import (for_intra)
from .util import load_args_data_pack


_DEFAULT_MODEL_ATTACH = "attach.model"
_DEFAULT_MODEL_RELATION = "relations.model"

# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


def _load_and_select_data(args):
    """
    read data and filter on fold if relevant
    """
    if args.fold is None:
        dpack = load_args_data_pack(args)
    else:
        # load data pack *AFTER* fold dict (fail faster)
        fold_dict = load_fold_dict(args.fold_file)
        dpack = load_args_data_pack(args)
        dpack = dpack.training(fold_dict, args.fold)

    if args.intrasentential:
        dpack = for_intra(dpack)

    return dpack


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------


def config_argparser(psr):
    "add subcommand arguments to subparser"

    add_common_args(psr)
    add_learner_args(psr)
    add_fold_choice_args(psr)
    psr.add_argument("--attachment-model", "-A", metavar="FILE",
                     default=_DEFAULT_MODEL_ATTACH,
                     help="save attachment model here "
                     "(default: %s)" % _DEFAULT_MODEL_ATTACH)
    psr.add_argument("--relation-model", "-R", metavar="FILE",
                     default=_DEFAULT_MODEL_RELATION,
                     help="save relation model here "
                     "(default: %s)" % _DEFAULT_MODEL_RELATION)
    psr.set_defaults(func=main)


@validate_fold_choice_args
@validate_learner_args
def main(args):
    "subcommand main (invoked from outer script)"

    dpack = _load_and_select_data(args)
    decoder = args_to_decoder(args)
    learners = args_to_learners(decoder, args)
    tasks = {Task.attach: args.attachment_model,
             Task.relate: args.relation_model}
    jobs = hlearn.jobs(dpack, learners, tasks, args.quiet)
    Parallel(n_jobs=-1)(jobs)
