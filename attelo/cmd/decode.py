"build a discourse graph from edu pairs and a model"

from __future__ import print_function
from functools import wraps
import csv
import json
import os
import sys

from ..args import\
    add_common_args, add_decoder_args,\
    add_fold_choice_args, validate_fold_choice_args,\
    args_to_decoder, args_to_phrasebook, args_to_threshold
from ..fold import folds_to_orange
from ..io import\
    read_data, load_model
from ..table import select_data_in_grouping
from ..decoding import\
    DecoderConfig, decode_document


NAME = 'decode'

# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


def _select_data(args, phrasebook):
    """
    read data into a pair of tables, filtering out training
    data if a fold is specified

    NB: in contrast, the learn._select_data filters out
    the test data
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
                                             args.fold)
        if data_relations:
            data_relations = data_relations.select_ref(selection,
                                                       args.fold)

    return data_attach, data_relations


def export_graph(predicted, doc, folder):
    """
    Write the graph out in an adhoc format, representing by
    lines of the form ::

        pred ( arg1 / arg2 )

    The output will be saved to FOLDER/DOC.rel
    """
    fname = os.path.join(folder, doc + ".rel")
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(fname, 'w') as fout:
        for arg1, arg2, rel in predicted:
            print("%s ( %s / %s )" % (rel, arg1, arg2),
                  file=fout)


def export_csv(phrasebook, predicted, doc, attach_instances, folder):
    """
    Write the predictions out as a CSV table in the Orange CSV
    format, with columns for the identifying meta phrasebook, and
    the assigned class.

    The output will be saved to FOLDER/DOC.csv
    """
    fname = os.path.join(folder, doc + ".csv")
    if not os.path.exists(folder):
        os.makedirs(folder)
    predicted_map = {(e1, e2): label for e1, e2, label in predicted}
    metas = attach_instances.domain.getmetas().values()

    with open(fname, 'wb') as fout:
        writer = csv.writer(fout)
        writer.writerow(["m#" + x.name for x in metas] +
                        ["c#" + phrasebook.label])
        for inst in attach_instances:
            du1 = inst[phrasebook.source].value
            du2 = inst[phrasebook.target].value
            label = predicted_map.get((du1, du2), "UNRELATED")
            writer.writerow([inst[x].value for x in metas] + [label])

# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------


def config_argparser(psr):
    "add subcommand arguments to subparser"

    add_common_args(psr)
    add_decoder_args(psr)
    add_fold_choice_args(psr)
    psr.add_argument("--attachment-model", "-A", default=None,
                     required=True,
                     help="model needed for attachment prediction")
    psr.add_argument("--relation-model", "-R", default=None,
                     help="model needed for relations prediction")
    psr.add_argument("--output", "-o",
                     default=None,
                     required=True,
                     metavar="DIR",
                     help="save predicted structures here")
    psr.set_defaults(func=main)


def validate_model_args(wrapped):
    """
    Given a function that accepts an argparsed object, check
    the model arguments before carrying on.

    Basically, the relation model is needed if you supply a
    relations file

    This is meant to be used as a decorator, eg.::

        @validate_fold_choice_args
        def main(args):
            blah
    """
    @wraps(wrapped)
    def inner(args):
        "die model args are incomplete"
        if args.data_relations and not args.relation_model:
            sys.exit("arg error: --relation-model is required if a "
                     "relation file is given")
        if args.relation_model and not args.data_relations:
            sys.exit("arg error: --relation-model is not needed "
                     "unless a relation data file is also given")
        wrapped(args)
    return inner


@validate_model_args
@validate_fold_choice_args
def main(args):
    "subcommand main"

    phrasebook = args_to_phrasebook(args)
    data_attach, data_relations = _select_data(args, phrasebook)
    # only one learner+decoder for now
    decoder = args_to_decoder(args)

    model_attach = load_model(args.attachment_model)
    model_relations = load_model(args.relation_model) if data_relations\
        else None

    threshold = args_to_threshold(model_attach,
                                  decoder,
                                  requested=args.threshold)

    config = DecoderConfig(phrasebook=phrasebook,
                           decoder=decoder,
                           threshold=threshold,
                           post_labelling=args.post_label,
                           use_prob=args.use_prob)

    grouping_index = data_attach.domain.index(phrasebook.grouping)
    all_groupings = frozenset(inst[grouping_index].value for
                              inst in data_attach)

    for onedoc in all_groupings:
        print("decoding on file : ", onedoc, file=sys.stderr)

        attach_instances, rel_instances =\
            select_data_in_grouping(phrasebook,
                                    onedoc,
                                    data_attach,
                                    data_relations)

        predicted = decode_document(config,
                                    model_attach, attach_instances,
                                    model_relations, rel_instances)
        export_graph(predicted, onedoc, args.output)
        export_csv(phrasebook, predicted, onedoc, attach_instances,
                   args.output)
