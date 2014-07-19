"build a discourse graph from edu pairs and a model"

from __future__ import print_function
import csv
import json
import os
import sys

from ..args import\
    add_common_args, add_decoder_args,\
    add_fold_choice_args, validate_fold_choice_args,\
    args_to_decoder, args_to_features, args_to_threshold
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


def _select_data(args, features):
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
                                    meta_index=features.grouping)
        data_attach = data_attach.select_ref(selection,
                                             args.fold)
        if data_relations:
            data_relations = data_relations.select_ref(selection,
                                                       args.fold)

    return data_attach, data_relations


def export_graph(predicted, doc, folder):
    fname = os.path.join(folder, doc + ".rel")
    if not os.path.exists(folder):
        os.makedirs(folder)
    f = open(fname, 'w')
    for (a1, a2, rel) in predicted:
        f.write(rel + " ( " + a1 + " / " + a2 + " )\n")
    f.close()


def export_csv(features, predicted, doc, attach_instances, folder):
    fname = os.path.join(folder, doc + ".csv")
    if not os.path.exists(folder):
        os.makedirs(folder)
    predicted_map = {(e1, e2): label for e1, e2, label in predicted}
    metas = attach_instances.domain.getmetas().values()

    with open(fname, 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(["m#" + x.name for x in metas] +
                        ["c#" + features.label])
        for r in attach_instances:
            row = [r[x].value for x in metas]
            e1 = r[features.source].value
            e2 = r[features.target].value
            epair = (e1, e2)
            label = predicted_map.get((e1, e2), "UNRELATED")
            writer.writerow(row + [label])

# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------


def config_argparser(psr):
    "add subcommand arguments to subparser"

    add_common_args(psr)
    add_decoder_args(psr)
    add_fold_choice_args(psr)
    psr.add_argument("--attachment-model", "-A", default=None,
                     help="provide saved model for prediction of "
                     "attachment (only with -T option)")
    psr.add_argument("--relation-model", "-R", default=None,
                     help="provide saved model for prediction of "
                     "relations (only with -T option)")
    psr.add_argument("--output", "-o",
                     default=None,
                     required=True,
                     metavar="DIR",
                     help="save predicted structures here")
    psr.set_defaults(func=main)


@validate_fold_choice_args
def main(args):
    "subcommand main"

    features = args_to_features(args)
    data_attach, data_relations = _select_data(args, features)
    # only one learner+decoder for now
    decoder = args_to_decoder(args)

    if not args.attachment_model:
        sys.exit("ERROR: [test mode] attachment model must be provided " +
                 "with -A")
    if data_relations and not args.relation_model:
        sys.exit("ERROR: [test mode] relation model must be provided if " +
                 "relation data is provided")

    model_attach = load_model(args.attachment_model)
    model_relations = load_model(args.relation_model) if data_relations\
        else None

    threshold = args_to_threshold(model_attach,
                                  decoder,
                                  requested=args.threshold)

    config = DecoderConfig(features=features,
                           decoder=decoder,
                           threshold=threshold,
                           post_labelling=args.post_label,
                           use_prob=args.use_prob)

    grouping_index = data_attach.domain.index(features.grouping)
    all_groupings = set()
    for inst in data_attach:
        all_groupings.add(inst[grouping_index].value)

    for onedoc in all_groupings:
        print("decoding on file : ", onedoc, file=sys.stderr)

        attach_instances, rel_instances =\
            select_data_in_grouping(features,
                                    onedoc,
                                    data_attach,
                                    data_relations)

        predicted = decode_document(config,
                                    model_attach, attach_instances,
                                    model_relations, rel_instances)
        export_graph(predicted, onedoc, args.output)
        export_csv(features, predicted, onedoc, attach_instances, args.output)
