"build a discourse graph from edu pairs and a model"

from __future__ import print_function
from collections import defaultdict
from functools import wraps
from itertools import chain
from os import path as fp
import argparse
import csv
import json
import os
import sys

from ..args import (add_common_args, add_decoder_args,
                    add_fold_choice_args, validate_fold_choice_args,
                    args_to_decoder, args_to_decoding_mode,
                    args_to_phrasebook)
from ..fold import folds_to_orange
from ..io import read_data, load_model
from ..table import (related_attachments, related_relations,
                     select_data_in_grouping)
from ..decoding import (DataAndModel, DecoderException,
                        decode, count_correct)
from ..report import Count


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


def select_fold(data_attach, data_relate, args, phrasebook):
    """
    Filter out training data according to what fold is specified
    in the args

    (potentially useful if you are writing your own attelo decode
    wrapper in your test harness)
    """

    fold_struct = json.load(args.fold_file)
    selection = folds_to_orange(data_attach,
                                fold_struct,
                                meta_index=phrasebook.grouping)
    data_attach = data_attach.select_ref(selection,
                                         args.fold)
    if data_relate is not None:
        data_relate = data_relate.select_ref(selection, args.fold)
    return data_attach, data_relate


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
        data_attach, data_relate =\
            select_fold(data_attach, data_relate,
                        args, phrasebook)
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


def select_doc(phrasebook, onedoc, attach, relate):
    """
    Given an attachment and relations data/model pair,
    return a narrower pair selecting only the data that
    correspond to a given document
    """
    relate_data = relate.data if relate is not None else None
    attach_instances, relate_instances =\
        select_data_in_grouping(phrasebook,
                                onedoc,
                                attach.data,
                                relate_data)
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


def _export_conllish(phrasebook, predicted, attach_instances, folder):
    """
    Append the predictions to our CONLL like output file documented in
    `doc/output.md`

    (FOLDER/graph.conll)
    """
    incoming = defaultdict(list)
    for edu1, edu2, label in predicted:
        incoming[edu2].append((edu1, label))
    max_indegree = max(len(x) for x in incoming.items()) if incoming else 1

    edus = {}
    for inst in attach_instances:
        edu1 = inst[phrasebook.source].value
        edu2 = inst[phrasebook.target].value
        edus[edu1] = (inst[phrasebook.source_span_start].value,
                      inst[phrasebook.source_span_end].value,
                      inst[phrasebook.grouping].value)
        edus[edu2] = (inst[phrasebook.target_span_start].value,
                      inst[phrasebook.target_span_end].value,
                      inst[phrasebook.grouping].value)

    def mk_row(edu):
        "csv row for the given edu"

        if edu == '0':
            raise ValueError('We assume that no EDU is labelled 0')

        start, end, grouping = edus[edu]
        if incoming.get(edu):
            linkstuff = list(chain.from_iterable(incoming[edu]))
        else:
            linkstuff = ["0", "ROOT"]
        pad_len = max_indegree * 2 - len(linkstuff)
        return [edu, grouping, start, end] + linkstuff + ([''] * pad_len)

    with open(fp.join(folder, 'graph.conll'), 'a') as fout:
        writer = csv.writer(fout, dialect=csv.excel_tab)
        for edu in sorted(edus, key=lambda x: x[0]):
            writer.writerow(mk_row(edu))


def _prepare_combined_outputs(folder):
    """
    Initialise any output files that are to be appended to rather
    than written separately
    """
    fname = os.path.join(folder, "graph.conll")
    if not os.path.exists(folder):
        os.makedirs(folder)
    open(fname, 'w').close()


def _write_predictions(phrasebook, doc, predicted, attach, output):
    """
    Save predictions to disk in various formats
    """
    _export_graph(predicted, doc, output)
    _export_conllish(phrasebook, predicted, attach.data, output)


def score_prediction(phrasebook, attach, relate, predicted):
    """
    Return the best prediction for the given data along with its
    score. Best is defined in a recall-centric way, by the number
    of correct labels made (or if in attach-only mode, the number
    of correct decisions to attach).

    :param relate: if True, labels (relations) are to be evaluated too
                   otherwise only attachments
    :param predicted: a single prediction (list of id, id, label tuples)
    """
    reference = related_attachments(phrasebook, attach.data)
    if relate:
        labels = related_relations(phrasebook, relate.data)
    else:
        labels = None
    return count_correct(phrasebook,
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
                     metavar='FILE',
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


def main_for_harness(args, phrasebook, decoder, attach, relate):
    """
    main function you can hook into if writing your own harness

    You have to supply DataModel args for attachment/relation
    yourself
    """
    _prepare_combined_outputs(args.output)
    if not attach.data:  # there may be legitimate uses for empty inputs
        return

    grouping_index = attach.data.domain.index(phrasebook.grouping)
    all_groupings = frozenset(inst[grouping_index].value for
                              inst in attach.data)

    scores = {}
    for onedoc in all_groupings:
        if not args.quiet:
            print("decoding on file : ", onedoc, file=sys.stderr)
        doc_attach, doc_relate = select_doc(phrasebook, onedoc, attach, relate)
        mode = args_to_decoding_mode(args)
        predictions = decode(phrasebook, mode, decoder,
                             doc_attach, doc_relate)
        if not predictions:
            raise DecoderException('decoder must make at least one prediction')

        # we trust the decoder to select what it thinks is its best prediction
        first_prediction = predictions[0]
        _write_predictions(phrasebook, onedoc, first_prediction,
                           doc_attach, args.output)
        if args.scores is not None:
            scores[onedoc] = score_prediction(phrasebook,
                                              doc_attach, doc_relate,
                                              first_prediction)

    if args.scores is not None:
        Count.write_csv(scores, args.scores)


@validate_model_args
@validate_fold_choice_args
def main(args):
    "subcommand main"

    phrasebook = args_to_phrasebook(args)
    attach, relate = _load_data_and_model(phrasebook, args)
    decoder = args_to_decoder(args)
    main_for_harness(args, phrasebook, decoder, attach, relate)
