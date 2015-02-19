"build a discourse graph from edu pairs and a model"

from __future__ import print_function
from os import path as fp
import json
import os

from joblib import (Parallel, delayed)

from ..args import (add_common_args, add_decoder_args,
                    add_model_read_args,
                    add_fold_choice_args, validate_fold_choice_args,
                    args_to_decoder, args_to_decoding_mode)
from ..io import (load_model, append_predictions_output)
from ..decoding import (DecoderException, decode, count_correct)
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
    add_model_read_args(psr, "model needed for {} prediction")
    psr.add_argument("--output", "-o",
                     default=None,
                     required=True,
                     metavar="FILE",
                     help="save predicted structures here")
    add_decoder_args(psr)
    add_fold_choice_args(psr)
    psr.set_defaults(func=main)


def load_models(args):
    '''
    Load model specified on the command line

    :rtype :py:class:Team:
    '''
    return Team(attach=load_model(args.attachment_model),
                relate=load_model(args.relation_model))


def _decode_group(mode, full_output, decoder, dpack, docset, models):
    '''
    decode a single group and write its output

    score the predictions if we have

    :param full_output output filename for combined outputs
    :param docset (current doc, all docs)

    :rtype Count or None
    '''
    onedoc, alldocs = docset
    output = tmp_output_filename(full_output, onedoc)
    predictions = decode(mode, decoder, dpack, models)
    if not predictions:
        raise DecoderException('decoder must make at least one prediction')

    # we trust the decoder to select what it thinks is its best prediction
    first_prediction = predictions[0]
    append_predictions_output(dpack, first_prediction, output)
    # ok we're done here for this document
    with open(output + '.done', 'wb'):
        pass
    _concatenate_files_if_done(full_output, alldocs)


def _concatenate_files_if_done(full_output, alldocs):
    """
    Check if we've completed all decoding tasks in our parallel
    set; concatenate the results.

    Return 'False' if we're only partially done: ie if some
    control files exist but not all
    """
    tmpfiles = [tmp_output_filename(full_output, d) for d in alldocs]
    if all(fp.exists(x + '.done') for x in tmpfiles):
        with open(full_output, 'wb') as file_out:
            for tfile in tmpfiles:
                with open(tfile, 'rb') as file_in:
                    file_out.write(file_in.read())
        _clean_temp_filenames(full_output, alldocs)
        return True
    elif any(fp.exists(x + '.done') for x in tmpfiles):
        return False
    else:
        return True


def _clean_temp_filenames(full_output, alldocs):
    """
    Remove temporary files that would be created during parallel
    decoding. This may be important if you interrupt a parallel
    decoding task
    """
    tmpfiles = [tmp_output_filename(full_output, d) for d in alldocs]
    for tmpfile in tmpfiles:
        done_file = tmpfile + '.done'
        if fp.exists(tmpfile):
            os.remove(tmpfile)
        if fp.exists(done_file):
            os.remove(done_file)


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
    combined output.

    We already try to do this at the end of each group-wide
    decoding, but this is necessary as race-condition
    proofing, in case of the corner-case where some jobs
    finish simultaneously and nobody thinks the other is done
    """
    alldocs = dpack.groupings().keys()
    status = _concatenate_files_if_done(args.output, alldocs)
    if not status:
        raise DecoderException('Found some but not all find temporary and '
                               'control files for parallel decoding; this '
                               'may be a bug; should be all or nothing')


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
    alldocs = groupings.keys()
    _clean_temp_filenames(args.output, alldocs)
    for onedoc, indices in groupings.items():
        onepack = dpack.selected(indices)
        jobs.append(delayed(_decode_group)(mode, args.output,
                                           decoder,
                                           onepack,
                                           (onedoc, alldocs),
                                           models))
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
    models = load_models(args)
    decoder = args_to_decoder(args)
    main_for_harness(args, decoder, dpack, models)
