"learn and save models"

from __future__ import print_function
import json

from ..args import\
    add_common_args, add_learner_args,\
    add_fold_choice_args, validate_fold_choice_args,\
    args_to_phrasebook, args_to_decoder, args_to_learners
from ..fold import folds_to_orange
from ..io import read_data, save_model, Torpor
from ..table import related_relations


NAME = 'learn'

_DEFAULT_MODEL_ATTACH = "attach.model"
_DEFAULT_MODEL_RELATION = "relations.model"

# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


def _select_data(args, phrasebook):
    """
    read data into a pair of tables, filtering out test
    data if a fold is specified
    """

    data_attach, data_relations =\
        read_data(args.data_attach, args.data_relations,
                  verbose=not args.quiet)
    if args.fold:
        fold_struct = json.load(args.fold_file)
        selection = folds_to_orange(data_attach,
                                    fold_struct,
                                    meta_index=phrasebook.grouping)
        data_attach = data_attach.select_ref(selection,
                                             args.fold,
                                             negate=1)
        if data_relations:
            data_relations = data_relations.select_ref(selection,
                                                       args.fold,
                                                       negate=1)

    return data_attach, data_relations

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
def main(args):
    "subcommand main (invoked from outer script)"

    phrasebook = args_to_phrasebook(args)
    data_attach, data_relations = _select_data(args, phrasebook)
    decoder = args_to_decoder(args)
    attach_learner, relation_learner = \
        args_to_learners(decoder, phrasebook, args)

    def torpor(msg):
        "training feedback"
        if args.fold:
            msg += " (fold %d)" % args.fold
        return Torpor(msg, sameline=False, quiet=args.quiet)

    with torpor("training attachment model"):
        model_attach = attach_learner(data_attach)
        save_model(args.attachment_model, model_attach)

    if data_relations:
        with torpor("training relations model"):
            related_only = related_relations(phrasebook, data_relations)
            model_relations = relation_learner(related_only)
            save_model(args.relation_model, model_relations)
