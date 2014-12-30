"run cross-fold evaluation"

from __future__ import print_function
import itertools
import json
import sys

from ..args import\
    add_common_args, add_learner_args, add_decoder_args,\
    add_report_args,\
    args_to_decoder,\
    args_to_phrasebook,\
    args_to_learners,\
    args_to_threshold
from ..decoding import\
    DataAndModel, DecoderConfig,\
    decode, count_correct
from ..fold import make_n_fold, folds_to_orange
from ..io import read_data
from ..report import Report
from ..table import\
    related_attachments, related_relations, select_data_in_grouping
# TODO: figure out what parts I want to export later
from .decode import\
    _select_doc,\
    _score_predictions,\
    _args_to_decoder_config


NAME = 'evaluate'


def _prepare_folds(phrasebook, num_folds, table, shuffle=True):
    """Return an N-fold validation setup respecting a property where
    examples in the same grouping stay in the same fold.
    """
    import random
    if shuffle:
        random.seed()
    else:
        random.seed("just an illusion")

    fold_struct = make_n_fold(table,
                              folds=num_folds,
                              meta_index=phrasebook.grouping)
    selection = folds_to_orange(table,
                                fold_struct,
                                meta_index=phrasebook.grouping)
    return fold_struct, selection


def config_argparser(psr):
    "add subcommand arguments to subparser"

    add_common_args(psr)
    add_learner_args(psr)
    add_report_args(psr)
    psr.set_defaults(func=main)
    psr.add_argument("--nfold", "-n",
                     default=10, type=int,
                     help="nfold cross-validation number (default 10)")
    psr.add_argument("-s", "--shuffle",
                     default=False, action="store_true",
                     help="if set, ensure a different cross-validation "
                     "of files is done, otherwise, the same file "
                     "splitting is done everytime")
    psr.add_argument("--unlabelled", "-u",
                     default=False, action="store_true",
                     help="force unlabelled evaluation, even if the "
                     "prediction is made with relations")


def main(args):
    phrasebook = args_to_phrasebook(args)
    data_attach, data_relate = read_data(args.data_attach,
                                         args.data_relations)
    # print(args, file=sys.stderr)
    decoder = args_to_decoder(args)
    
    # TODO: more models for intra-sentence
    attach_learner, relation_learner = \
        args_to_learners(decoder, phrasebook, args)

    RECALL_CORRECTION = args.correction

    fold_struct, selection =\
        _prepare_folds(phrasebook, args.nfold, data_attach,
                       shuffle=args.shuffle)

    with_relations = bool(data_relate)
    args.relations = ["attach", "relations"][with_relations]
    args.context = "window5" if "window" in args.data_attach else "full"

    # eval procedures
    score_labels = with_relations and not args.unlabelled

    evals = []
    # --- fold level -- to be refactored
    for test_fold in range(args.nfold):
        print(">>> doing fold ", test_fold + 1, file=sys.stderr)
        print(">>> training ... ", file=sys.stderr)

        train_data_attach = data_attach.select_ref(selection,
                                                   test_fold,
                                                   negate=1)
        # train model
        # TODO: separate models for intra-sentence/inter-sentence
        model_attach = attach_learner(train_data_attach)

        if with_relations:
            train_data_relate = data_relate.select_ref(selection,
                                                       test_fold,
                                                       negate=1)
            train_data_relate = related_relations(phrasebook,
                                                  train_data_relate)
            # train model
            model_relate = relation_learner(train_data_relate)
        else:  # no relations
            model_relate = None

        attach = DataAndModel(data_attach, model_attach)
        relate = DataAndModel(data_relate, model_relate)\
            if data_relate else None

        # decoding options for this fold
        config = _args_to_decoder_config(phrasebook,
                                         attach.model,
                                         decoder,
                                         args)

        # -- file level --
        fold_evals = []
        for onedoc in fold_struct:
            if fold_struct[onedoc] == test_fold:
                print("decoding on file : ", onedoc, file=sys.stderr)

                doc_attach, doc_relate =\
                    _select_doc(config, onedoc, attach, relate)
                predicted = decode(config, decoder, doc_attach, doc_relate,
                                   nbest=args.nbest)

                score_doc_relate = doc_relate if score_labels else None
                fold_evals.append(_score_predictions(config,
                                                     doc_attach,
                                                     score_doc_relate,
                                                     predicted,
                                                     nbest=args.nbest,))

        fold_report = Report(fold_evals,
                             params=args,
                             correction=RECALL_CORRECTION)
        print("Fold eval:", fold_report.summary())
        evals.append(fold_evals)
        # --end of file level
       # --- end of fold level
    # end of test for a set of parameter
    report = Report(list(itertools.chain.from_iterable(evals)),
                    params=args,
                    correction=args.correction)
    print(">>> FINAL EVAL:", report.summary())
