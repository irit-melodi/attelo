"learn and save models"

from __future__ import print_function
import json

from ..args import\
    (add_common_args,
     add_learner_args, validate_learner_args,
     add_fold_choice_args, validate_fold_choice_args,
     args_to_decoder, args_to_learners)
from ..io import load_data_pack, save_model, Torpor
from ..learning import learn
from ..table import for_attachment, for_labelling


_DEFAULT_MODEL_ATTACH = "attach.model"
_DEFAULT_MODEL_RELATION = "relations.model"

# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


def _load_and_select_data(args):
    """
    read data and filter on fold if relevant
    """
    dpack = load_data_pack(args.edus, args.features,
                           verbose=not args.quiet)
    if args.fold is None:
        return dpack
    else:
        fold_dict = json.load(args.fold_file)
        return dpack.training(fold_dict, args.fold)

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


def main_for_harness(args, dpack):
    """
    main function core that you can hook into if writing your own
    harness

    You have to supply (and filter) the data yourself
    (see `select_data`)
    """
    decoder = args_to_decoder(args)
    learners = args_to_learners(decoder, args)
    models = learn(learners, dpack, verbose=True)
    with Torpor('writing models'):
        save_model(args.attachment_model, models.attach)
        save_model(args.relation_model, models.relate)


@validate_fold_choice_args
@validate_learner_args
def main(args):
    "subcommand main (invoked from outer script)"

    dpack = _load_and_select_data(args)
    main_for_harness(args, dpack)
