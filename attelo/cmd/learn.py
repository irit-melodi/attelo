"learn and save models"

from __future__ import print_function
import sys

from ..args import\
    add_common_args, add_decoder_args, add_learner_args,\
    args_to_features, args_to_decoder, args_to_learners
from ..io import read_data, save_model, Torpor
from ..table import related_relations


NAME = 'learn'


def config_argparser(psr):
    "add subcommand arguments to subparser"

    add_common_args(psr)
    add_learner_args(psr)
    psr.add_argument("--attachment-model", "-A",
                     default="attach.model",
                     help="save attachment model here")
    psr.add_argument("--relation-model", "-R",
                     default="relations.model",
                     help="save relation model here")
    psr.set_defaults(func=main)


def main(args):
    "subcommand main (invoked from outer script)"

    data_attach, data_relations =\
        read_data(args.data_attach, args.data_relations,
                  verbose=not args.quiet)
    features = args_to_features(args)
    decoder = args_to_decoder(args)
    attach_learner, relation_learner = \
        args_to_learners(decoder, features, args)

    torpor = lambda msg: Torpor(msg, sameline=False, quiet=args.quiet)

    with torpor("training attachment model"):
        model_attach = attach_learner(data_attach)
        save_model(args.attachment_model, model_attach)

    if data_relations:
        with torpor("training relations model"):
            related_only = related_relations(features, data_relations)
            model_relations = relation_learner(related_only)
            save_model(args.relation_model, model_relations)
