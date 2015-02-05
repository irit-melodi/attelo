"build a discourse graph from edu pairs and a model"

from __future__ import print_function
import json
import sys

from ..args import (add_common_args, add_decoder_args,
                    add_fold_choice_args, validate_fold_choice_args,
                    args_to_decoder, args_to_decoding_mode)
from ..io import (load_model,
                  start_predictions_output,
                  append_predictions_output)
from ..decoding import (DecoderException, decode, count_correct)
from ..report import Count
from ..util import Team
from .util import load_args_data_pack


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


def _load_and_select_data(args):
    """
    read data and filter on fold if relevant
    """
    dpack = load_args_data_pack(args)
    if args.fold is None:
        return dpack
    else:
        fold_dict = json.load(args.fold_file)
        return dpack.testing(fold_dict, args.fold)


def score_prediction(dpack, predicted):
    """
    Return the best prediction for the given data along with its
    score. Best is defined in a recall-centric way, by the number
    of correct labels made (or if in attach-only mode, the number
    of correct decisions to attach).

    :param predicted: a single prediction (list of id, id, label tuples)
    """
    return count_correct(dpack.attached_only(), predicted)

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
                     required=True,
                     help="model needed for relations prediction")
    psr.add_argument("--scores",
                     metavar='FILE',
                     help="score our decoding (test data must have "
                     "ref labels to score against) and save it here")
    psr.add_argument("--output", "-o",
                     default=None,
                     required=True,
                     metavar="FILE",
                     help="save predicted structures here")
    psr.set_defaults(func=main)


def load_models(args):
    '''
    Load model specified on the command line

    :rtype :py:class:Team:
    '''
    return Team(attach=load_model(args.attachment_model),
                relate=load_model(args.relation_model))


def _decode_group(args, decoder, dpack, models):
    '''
    decode a single group and write its output

    score the predictions if we have

    :rtype Count or None
    '''
    mode = args_to_decoding_mode(args)
    predictions = decode(mode, decoder, dpack, models)
    if not predictions:
        raise DecoderException('decoder must make at least one prediction')

    # we trust the decoder to select what it thinks is its best prediction
    first_prediction = predictions[0]
    append_predictions_output(dpack, first_prediction,
                              args.output)
    if args.scores:
        return score_prediction(dpack, first_prediction)
    else:
        return None


def main_for_harness(args, decoder, dpack, models):
    """
    main function you can hook into if writing your own harness

    You have to supply DataModel args for attachment/relation
    yourself
    """
    start_predictions_output(args.output)
    groupings = dpack.groupings()

    scores = {}
    for onedoc, indices in groupings.items():
        if not args.quiet:
            print("decoding on file : ", onedoc, file=sys.stderr)
        onepack = dpack.selected(indices)
        scores[onedoc] = _decode_group(args, decoder, onepack, models)

    if args.scores is not None:
        with open(args.scores, 'w') as stream:
            Count.write_csv(scores, stream)


@validate_fold_choice_args
def main(args):
    "subcommand main"
    dpack = _load_and_select_data(args)
    models = load_models(args)
    decoder = args_to_decoder(args)
    main_for_harness(args, decoder, dpack, models)
