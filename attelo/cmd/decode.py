"build a discourse graph from edu pairs and a model"

from __future__ import print_function
from os import path as fp
import os

from joblib import (Parallel, delayed)

from ..args import (add_common_args, add_decoder_args,
                    add_model_read_args,
                    add_fold_choice_args, validate_fold_choice_args,
                    args_to_decoder, args_to_decoding_mode)
from ..io import (load_model, load_fold_dict,
                  write_predictions_output)
from ..decoding import (DecoderException, decode)
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
        return dpack
    else:
        # load fold dictionary before data pack
        # this way, if it fails we find out sooner
        # instead of waiting for the data pack
        fold_dict = load_fold_dict(args.fold_file)
        dpack = load_args_data_pack(args)
        return dpack.testing(fold_dict, args.fold)


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------


def config_argparser(psr):
    "add subcommand arguments to subparser"

    add_common_args(psr)
    add_model_read_args(psr, "model needed for {} prediction")
    psr.add_argument("--output", "-o",
                     default=None,
                     required=True,
                     metavar="FILE",
                     help="save predicted structures here")
    add_decoder_args(psr)
    add_fold_choice_args(psr)
    psr.set_defaults(func=main)


def load_models(paths):
    '''
    Load model specified on the command line

    :type: paths: Team(string)

    :rtype :py:class:Team:
    '''
    return Team(attach=load_model(paths.attach),
                relate=load_model(paths.relate))


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
    write_predictions_output(dpack, first_prediction, output)


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
    tmpfiles = [tmp_output_filename(args.output, d)
                for d in groupings]
    for tmpfile in tmpfiles:
        if fp.exists(tmpfile):
            os.remove(tmpfile)
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
    Parallel(n_jobs=-1,
             verbose=5)(delayed_main_for_harness(args, decoder, dpack, models))
    concatenate_outputs(args, dpack)


@validate_fold_choice_args
def main(args):
    "subcommand main"
    dpack = _load_and_select_data(args)
    models = load_models(Team(attach=args.attachment_model,
                              relate=args.relation_model))
    decoder = args_to_decoder(args)
    main_for_harness(args, decoder, dpack, models)
