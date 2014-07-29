"build a discourse graph from edu pairs and a model"

from __future__ import print_function
from functools import wraps
import argparse
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
from ..table import\
    related_attachments, related_relations, select_data_in_grouping
from ..decoding import\
    DataAndModel, DecoderConfig, decode, count_correct
from ..report import Count


NAME = 'decode'

# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


def _args_to_decoder_config(phrasebook, model, decoder, args):
    """
    Package up command line arguments into a DecoderConfig
    """
    threshold = args_to_threshold(model, decoder,
                                  requested=args.threshold)
    return DecoderConfig(phrasebook=phrasebook,
                         threshold=threshold,
                         post_labelling=args.post_label,
                         use_prob=args.use_prob)


def _select_data(args, phrasebook):
    """
    read data into a pair of tables, filtering out training
    data if a fold is specified

    NB: in contrast, the learn._select_data filters out
    the test data
    """

    data_attach, data_relate =\
        read_data(args.data_attach, args.data_relations,
                  verbose=not args.quiet)
    if args.fold is not None:
        fold_struct = json.load(args.fold_file)
        selection = folds_to_orange(data_attach,
                                    fold_struct,
                                    meta_index=phrasebook.grouping)
        data_attach = data_attach.select_ref(selection,
                                             args.fold)
        if data_relate is not None:
            data_relate = data_relate.select_ref(selection, args.fold)

    return data_attach, data_relate


def _load_data_and_model(phrasebook, args):
    """
    Return DataAndModel pair for attachments and relations
    """
    data_attach, data_relate = _select_data(args, phrasebook)
    model_attach = load_model(args.attachment_model)
    attach = DataAndModel(data_attach, model_attach)
    if data_relate is not None:
        model_relate = load_model(args.relation_model)
        relate = DataAndModel(data_relate, model_relate)
    else:
        relate = None
    return attach, relate


def _select_doc(config, onedoc, attach, relate):
    """
    Given an attachment and relations data/model pair,
    return a narrower pair selecting only the data that
    correspond to a given document
    """
    attach_instances, relate_instances =\
        select_data_in_grouping(config.phrasebook,
                                onedoc,
                                attach.data,
                                relate.data)
    doc_attach = DataAndModel(attach_instances, attach.model)
    doc_relate = DataAndModel(relate_instances, relate.model)\
        if relate else None
    return doc_attach, doc_relate


def _export_graph(predicted, doc, folder):
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


def _export_csv(phrasebook, doc, predicted, attach_instances, folder):
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


def _write_predictions(config, doc, predicted, attach, output):
    """
    Save predictions to disk in various formats
    """
    _export_graph(predicted, doc, output)
    _export_csv(config.phrasebook, doc, predicted, attach.data, output)


def _score_predictions(config, attach, relate, predicted):
    """
    Return scores for predictions on the given data
    """
    reference = related_attachments(config.phrasebook, attach.data)
    labels = related_relations(config.phrasebook, relate.data)\
        if relate else None
    return count_correct(config.phrasebook,
                         predicted,
                         reference,
                         labels=labels)


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
    psr.add_argument("--scores",
                     type=argparse.FileType('w'),
                     help="score our decoding (test data must have "
                     "ref labels to score against) and save it here")
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
        if args.data_relations is not None and args.relation_model is None:
            sys.exit("arg error: --relation-model is required if a "
                     "relation file is given")
        if args.relation_model is not None and args.data_relations is None:
            sys.exit("arg error: --relation-model is not needed "
                     "unless a relation data file is also given")
        wrapped(args)
    return inner


@validate_model_args
@validate_fold_choice_args
def main(args):
    "subcommand main"

    phrasebook = args_to_phrasebook(args)
    decoder = args_to_decoder(args)
    attach, relate = _load_data_and_model(phrasebook, args)
    config = _args_to_decoder_config(phrasebook,
                                     attach.model,
                                     decoder,
                                     args)

    grouping_index = attach.data.domain.index(phrasebook.grouping)
    all_groupings = frozenset(inst[grouping_index].value for
                              inst in attach.data)

    scores = {}
    for onedoc in all_groupings:
        if not args.quiet:
            print("decoding on file : ", onedoc, file=sys.stderr)
        doc_attach, doc_relate = _select_doc(config, onedoc, attach, relate)
        predicted = decode(config, decoder, doc_attach, doc_relate)
        _write_predictions(config, onedoc, predicted, doc_attach, args.output)
        if args.scores is not None:
            scores[onedoc] = _score_predictions(config, doc_attach, doc_relate,
                                                predicted)
    if args.scores is not None:
        Count.write_csv(scores, args.scores)
