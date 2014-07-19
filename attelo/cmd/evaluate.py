"run cross-fold evaluation"

from __future__ import print_function
import itertools
import json
import sys

from ..args import\
    add_common_args, add_learner_args, add_decoder_args,\
    args_to_decoder,\
    args_to_features,\
    args_to_learners,\
    args_to_threshold
from ..decoding import DecoderConfig, decode_document
from ..fold import make_n_fold, folds_to_orange
from ..io import read_data
from ..report import Report
from ..table import\
    related_attachments, related_relations,\
    select_data_in_grouping


NAME = 'evaluate'


def _prepare_folds(features, num_folds, table, shuffle=True):
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
                              meta_index=features.grouping)
    selection = folds_to_orange(table,
                                fold_struct,
                                meta_index=features.grouping)
    return fold_struct, selection


def _discourse_eval(features,
                    predicted, reference,
                    labels=None, debug=False):
    """basic eval: counting correct predicted edges (labelled or not)
    data contains the reference attachments
    labels the corresponding relations
    """
    #print("REF:", reference)
    #print("PRED:", predicted)
    score = 0
    dict_predicted = dict([((a1, a2), rel) for (a1, a2, rel) in predicted])
    for one in reference:
        arg1 = one[features.source].value
        arg2 = one[features.target].value
        if debug:
            print(arg1, arg2, dict_predicted.get((arg1, arg2)),
                  file=sys.stderr)
        if (arg1, arg2) in dict_predicted:
            if labels is None:
                score += 1
                if debug:
                    print("correct", file=sys.stderr)
            else:
                relation_ref = labels.filter_ref({features.source: [arg1],
                                                  features.target: [arg2]})
                if len(relation_ref) == 0:
                    print("attached pair without corresponding relation",
                          one[features.grouping], arg1, arg2,
                          file=sys.stderr)
                else:
                    relation_ref = relation_ref[0][features.label].value
                    score += (dict_predicted[(arg1, arg2)] == relation_ref)

    total_ref = len(reference)
    total_pred = len(predicted)
    return score, total_pred, total_ref


def _save_scores(evals, args):
    """
    Save results of crossfold evaluation (list of list of individual
    scores)
    """
    report = Report(list(itertools.chain.from_iterable(evals)),
                    params=args,
                    correction=args.correction)
    json_scores = []
    for fold_evals in evals:
        fold_report = Report(fold_evals,
                             params=args,
                             correction=args.correction)
        json_scores.append(fold_report.json_scores())

    json_report = {"params": report.json_params(),
                   "combined_scores": report.json_scores(),
                   "fold_scores": json_scores}
    print(">>> FINAL EVAL:", report.summary())
    fname_fmt = "_".join("{" + x + "}" for x in
                         ["relations",
                          "context",
                          "decoder",
                          "learner",
                          "relation_learner",
                          "heuristics",
                          "unlabelled",
                          "post_label",
                          "rfc"])
    report.save("results/"+fname_fmt.format(**args.__dict__))
    with open("results/"+fname_fmt.format(**args.__dict__) + ".json", "wb")\
            as fout:
        json.dump(json_report, fout, indent=2)


def config_argparser(psr):
    "add subcommand arguments to subparser"

    add_common_args(psr)
    add_learner_args(psr)
    psr.set_defaults(func=main)
    psr.add_argument("--nfold", "-n",
                     default=10, type=int,
                     help="nfold cross-validation number (default 10)")
    psr.add_argument("-s", "--shuffle",
                     default=False, action="store_true",
                     help="if set, ensure a different cross-validation "
                     "of files is done, otherwise, the same file "
                     "splitting is done everytime")
    psr.add_argument("--correction", "-c",
                     default=1.0, type=float,
                     help="if input is already a restriction on the full "
                     "task, this options defines a correction to apply "
                     "on the final recall score to have the real scores "
                     "on the full corpus")
    psr.add_argument("--unlabelled", "-u",
                     default=False, action="store_true",
                     help="force unlabelled evaluation, even if the "
                     "prediction is made with relations")
    psr.add_argument("--accuracy", "-a",
                     default=False, action="store_true",
                     help="provide accuracy scores for classifiers used")


def main(args):
    features = args_to_features(args)
    data_attach, data_relations = read_data(args.data_attach,
                                            args.data_relations)

    decoder = args_to_decoder(args)
    attach_learner, relation_learner = \
        args_to_learners(decoder, features, args)

    RECALL_CORRECTION = args.correction

    fold_struct, selection =\
        _prepare_folds(features, args.nfold, data_attach,
                       shuffle=args.shuffle)

    with_relations = bool(data_relations)
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
        model_attach = attach_learner(train_data_attach)

        # test
        test_data_attach = data_attach.select_ref(selection, test_fold)

        if with_relations:
            train_data_relations = data_relations.select_ref(selection,
                                                             test_fold,
                                                             negate=1)
            train_data_relations = related_relations(features,
                                                     train_data_relations)
            # train model
            model_relations = relation_learner(train_data_relations)
        else:  # no relations
            model_relations = None

        # decoding options for this fold
        threshold = args_to_threshold(model_attach,
                                      decoder,
                                      requested=args.threshold)
        config = DecoderConfig(features=features,
                               decoder=decoder,
                               threshold=threshold,
                               post_labelling=args.post_label,
                               use_prob=args.use_prob)

        # -- file level --
        fold_evals = []
        for onedoc in fold_struct:
            if fold_struct[onedoc] == test_fold:
                print("decoding on file : ", onedoc, file=sys.stderr)

                attach_instances, rel_instances = \
                    select_data_in_grouping(features,
                                            onedoc,
                                            data_attach,
                                            data_relations)

                predicted = decode_document(config,
                                            model_attach, attach_instances,
                                            model_relations, rel_instances)

                reference = related_attachments(features, attach_instances)
                labels =\
                    related_relations(features, rel_instances) if score_labels\
                    else None
                scores = _discourse_eval(features,
                                         predicted,
                                         reference,
                                         labels=labels)
                fold_evals.append(scores)

        fold_report = Report(fold_evals,
                             params=args,
                             correction=RECALL_CORRECTION)
        print("Fold eval:", fold_report.summary())
        evals.append(fold_evals)
        # --end of file level
       # --- end of fold level
    # end of test for a set of parameter
    _save_scores(evals, args)
