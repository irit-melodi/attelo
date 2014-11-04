"build a discourse graph from edu pairs and a model"

from __future__ import print_function
from collections import defaultdict
from functools import wraps
from itertools import chain, repeat
import argparse
import itertools
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


def _export_conllish(phrasebook, doc, predicted, attach_instances, folder):
    """
    Append the predictions to our CONLL like output file documented in
    `doc/output.md`

    (FOLDER/graph.conll)
    """
    fname = os.path.join(folder, "graph.conll")
    incoming = defaultdict(list)
    for edu1, edu2, label in predicted:
        incoming[edu2].append((edu1, label))
    max_indegree = max(len(x) for x in incoming.items())

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

    concat_map = lambda f, xs: list(chain.from_iterable(map(f, xs)))

    def mk_row(edu):
        "csv row for the given edu"

        if edu == '0':
            raise ValueError('We assume that no EDU is labelled 0')

        start, end, grouping = edus[edu]
        row = [edu, grouping, start, end]
        linkstuff = []
        if incoming.get(edu):
            linkstuff = concat_map(list, incoming[edu])
        else:
            linkstuff = ["0", "ROOT"]
        pad_len = max_indegree * 2 - len(linkstuff)
        return row + linkstuff + ([''] * pad_len)


    with open(fname, 'a') as fout:
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
    open(fname, 'a').close()


def _write_predictions(config, doc, predicted, attach, output):
    """
    Save predictions to disk in various formats
    """
    _export_graph(predicted, doc, output)
    _export_csv(config.phrasebook, doc, predicted, attach.data, output)
    _export_conllish(config.phrasebook, doc, predicted, attach.data, output)


def _score_predictions(config, attach, relate, predicted,nbest=1):
    """
    Return scores for predictions on the given data

    'relate' if True, labels (relations) are to be evaluated to, otherwise only attachments 

    nbest=1: plain score for 1 prediction
    nbest: 'predicted' is an ordered list, returns the best score on the set of predictions
               best here means attachment score if relate=False, relations score otherwise 
    """
    reference = related_attachments(config.phrasebook, attach.data)
    labels = related_relations(config.phrasebook, relate.data)\
        if relate else None
    if nbest>1:
        all_counts = [count_correct(config.phrasebook,
                                    one_predicted,
                                    reference,
                                    labels=labels)
                      for one_predicted in predicted]
        # count the best relation score or the best attachment is there is no relation labels
        return max(all_counts,key=lambda x : x.correct_label) if labels else max(all_counts,key=lambda x: x.correct_attach) 
    else:
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


def main_for_harness(args, config, decoder, attach, relate):
    """
    main function you can hook into if writing your own harness

    You have to supply DataModel args for attachment/relation
    yourself
    """
    grouping_index = attach.data.domain.index(config.phrasebook.grouping)
    all_groupings = frozenset(inst[grouping_index].value for
                              inst in attach.data)

    scores = {}
    _prepare_combined_outputs(args.output)
    for onedoc in all_groupings:
        if not args.quiet:
            print("decoding on file : ", onedoc, file=sys.stderr)
        doc_attach, doc_relate = _select_doc(config, onedoc, attach, relate)
        predicted = decode(config, decoder, doc_attach, doc_relate,nbest=args.nbest)
        _write_predictions(config, onedoc, predicted, doc_attach, args.output)
        if args.scores is not None:
            scores[onedoc] = _score_predictions(config, doc_attach, doc_relate,
                                                predicted)
    if args.scores is not None:
        Count.write_csv(scores, args.scores)


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
    main_for_harness(args, config, decoder, attach, relate)
