"run cross-fold evaluation"

from __future__ import print_function
import itertools
import sys

from ..args import\
    (add_common_args, add_learner_args,
     add_report_args,
     args_to_decoder,
     args_to_decoding_mode,
     args_to_phrasebook,
     args_to_learners)
from ..decoding import\
    (DataAndModel, decode)
from ..fold import make_n_fold, folds_to_orange
from ..io import read_data
from ..report import Report
from .decode import select_doc, score_prediction


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


def _build_model_for_fold(selection, test_fold, learner, data):
    '''
    Return models for the training data in the given folds,
    packaging them up with their data set

    :rtype DataAndModel
    '''
    # by rights this should really select a test set but we
    # don't bother because we filter on it later on anyway
    # to pick out individual docs,
    train_data = data.select_ref(selection, test_fold, negate=1)
    return DataAndModel(data, learner(train_data))


def best_prediction(phrasebook, attach, relate, predictions):
    """
    Return the best prediction for the given data along with its
    score. Best is defined in a recall-centric way, by the number
    of correct labels made (or if in attach-only mode, the number
    of correct decisions to attach).

    :param relate: if True, labels (relations) are to be evaluated too
                   otherwise only attachments
    :param predicted: a single prediction (list of id, id, label tuples)
    """
    def score(prediction):
        'score a single prediction'
        return score_prediction(phrasebook, attach, relate, prediction)

    max_key = lambda x: score(x).correct_label if relate\
                        else score(x).correct_attach
    return max(predictions, key=max_key)


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
    'subcommand main'

    phrasebook = args_to_phrasebook(args)
    data_attach, data_relate = read_data(args.data_attach,
                                         args.data_relations)
    # print(args, file=sys.stderr)
    decoder = args_to_decoder(args)
    decoding_mode = args_to_decoding_mode(args)

    # TODO: more models for intra-sentence
    attach_learner, relation_learner = \
        args_to_learners(decoder, phrasebook, args)

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

        # train model
        # TODO: separate models for intra-sentence/inter-sentence
        attach = _build_model_for_fold(selection,
                                       test_fold,
                                       attach_learner,
                                       data_attach)
        relate = _build_model_for_fold(selection,
                                       test_fold,
                                       relation_learner,
                                       data_relate)\
                if with_relations else None

        # -- file level --
        fold_evals = []
        for onedoc in fold_struct:
            if fold_struct[onedoc] == test_fold:
                print("decoding on file : ", onedoc, file=sys.stderr)

                doc_attach, doc_relate =\
                    select_doc(phrasebook, onedoc, attach, relate)
                predictions = decode(phrasebook, decoding_mode, decoder,
                                     doc_attach, doc_relate)
                best = best_prediction(phrasebook, attach, relate,
                                       predictions)

                score_doc_relate = doc_relate if score_labels else None
                fold_evals.append(score_prediction(phrasebook,
                                                   doc_attach,
                                                   score_doc_relate,
                                                   best))

        fold_report = Report(fold_evals,
                             params=args,
                             correction=args.correction)
        print("Fold eval:", fold_report.summary())
        evals.append(fold_evals)
        # --end of file level
    # --- end of fold level
    # end of test for a set of parameter
    report = Report(list(itertools.chain.from_iterable(evals)),
                    params=args,
                    correction=args.correction)
    print(">>> FINAL EVAL:", report.summary())
