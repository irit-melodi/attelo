"run cross-fold evaluation"

from __future__ import print_function
import itertools
import sys

from ..args import (add_common_args,
                    add_learner_args, validate_learner_args,
                    add_report_args,
                    args_to_decoder,
                    args_to_decoding_mode,
                    args_to_learners)
from ..decoding import (decode)
from ..learning import (learn)
from ..fold import (make_n_fold, select_training, select_testing)
from ..report import EdgeReport
from ..score import (score_edges)
from ..util import (mk_rng)
from .util import load_args_multipack


def best_prediction(dpack, predictions):
    """
    Return the best prediction for the given data along with its
    score. Best is defined in a recall-centric way, by the number
    of correct labels made (or if in attach-only mode, the number
    of correct decisions to attach).

    :param relate: if True, labels (relations) are to be evaluated too
                   otherwise only attachments
    :param predicted: a single prediction (list of id, id, label tuples)
    """
    max_key = lambda x: score_edges(dpack, x).tpos_label
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


def _decode_group(dpack, models, decoder, mode):
    '''
    decode and score a single group

    :rtype Count
    '''
    predictions = decode(dpack, models, decoder, mode)
    best = best_prediction(dpack, predictions)
    return score_edges(dpack, best)


def _decode_fold(mpack, models, decoder, mode):
    '''
    decode and score all groups in the pack
    (pack should be whittled down to test set for
    a given fold)

    :rtype [Count]
    '''
    scores = []
    for onedoc, dpack in mpack.items():
        print("decoding on file : ", onedoc, file=sys.stderr)
        score = _decode_group(dpack, models, decoder, mode)
        scores.append(score)
    return scores


@validate_learner_args
def main(args):
    'subcommand main'

    mpack = load_args_multipack(args)
    # print(args, file=sys.stderr)
    decoder = args_to_decoder(args)
    decoding_mode = args_to_decoding_mode(args)

    # TODO: more models for intra-sentence
    learners = args_to_learners(decoder, args)

    fold_dict = make_n_fold(mpack.keys(), args.nfold,
                            mk_rng(args.shuffle))

    evals = []
    # --- fold level -- to be refactored
    for fold in range(args.nfold):
        print(">>> doing fold ", fold + 1, file=sys.stderr)
        print(">>> training ... ", file=sys.stderr)

        models = learn(select_training(mpack, fold_dict, fold),
                       learners)
        fold_evals = _decode_fold(select_testing(mpack, fold_dict, fold),
                                  models, decoder,
                                  decoding_mode)
        fold_report = EdgeReport(fold_evals,
                                 params=args,
                                 correction=args.correction)
        print("Fold eval:", fold_report.summary())
        evals.append(fold_evals)
        # --end of file level
    # --- end of fold level
    # end of test for a set of parameter
    report = EdgeReport(list(itertools.chain.from_iterable(evals)),
                        params=args,
                        correction=args.correction)
    print(">>> FINAL EVAL:", report.summary())
