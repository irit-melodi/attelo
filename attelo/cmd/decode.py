"build a discourse graph from edu pairs and a model"

from __future__ import print_function
from os import path as fp
import json
import os
import sys

from joblib import (Parallel, delayed)

from ..args import (add_common_args, add_decoder_args,
                    add_fold_choice_args, validate_fold_choice_args,
                    args_to_decoder, args_to_decoding_mode)
from ..io import (load_model, append_predictions_output)
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
    if args.fold is None:
        dpack = load_args_data_pack(args)
        return load_args_data_pack(args)
    else:
        # load fold dictionary before data pack
        # this way, if it fails we find out sooner
        # instead of waiting for the data pack
        fold_dict = json.load(args.fold_file)
        dpack = load_args_data_pack(args)
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


def _decode_group(mode, output, decoder, dpack, models):
    '''
    decode a single group and write its output

    score the predictions if we have

    :rtype Count or None
    '''
    predictions = decode(mode, decoder, dpack, models)
    if not predictions:
        raise DecoderException('decoder must make at least one prediction')

    # we trust the decoder to select what it thinks is its best prediction
    first_prediction = predictions[0]
    append_predictions_output(dpack, first_prediction, output)


def tmp_output_filename(path, suffix):
    """
    Temporary filename for output file segment
    """
    return fp.join(fp.dirname(path),
                   '_' + fp.basename(path) + '.' + suffix)


def concatenate_outputs(args, dpack):
    """
    (For use after :py:func:`delayed_main_for_harness`)

    Concatenate temporary per-group outputs into a single
    combined output
    """
    groupings = dpack.groupings()
    tmpfiles = [tmp_output_filename(args.output, d)
                for d in groupings]
    with open(args.output, 'wb') as file_out:
        for tfile in tmpfiles:
            with open(tfile, 'rb') as file_in:
                file_out.write(file_in.read())
    for tmpfile in tmpfiles:
        os.remove(tmpfile)


def delayed_main_for_harness(args, decoder, dpack, models):
    """
    Advanced variant of the main function which returns a list
    of decoding futures, each corresponding to the decoding
    task for a single group.

    Unlike the normal main function, this writes to a separate
    file for each grouping. It's up to you to concatenate the
    results after the fact
    """
    groupings = dpack.groupings()
    mode = args_to_decoding_mode(args)
    jobs = []
    for onedoc, indices in groupings.items():
        onepack = dpack.selected(indices)
        output = tmp_output_filename(args.output, onedoc)
        jobs.append(delayed(_decode_group)(mode, output,
                                           decoder, onepack, models))
    return jobs


def main_for_harness(args, decoder, dpack, models):
    """
    main function you can hook into if writing your own harness

    You have to supply DataModel args for attachment/relation
    yourself
    """
    Parallel(n_jobs=-1, verbose=5)(delayed_main_for_harness(args, decoder, dpack, models))
    concatenate_outputs(args, dpack)


@validate_fold_choice_args
def main(args):
    "subcommand main"
    dpack = _load_and_select_data(args)
    models = load_models(args)
    decoder = args_to_decoder(args)
    main_for_harness(args, decoder, dpack, models)
