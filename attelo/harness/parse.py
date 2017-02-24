'''
Control over attelo parsers as might be needed for a test harness
'''

from __future__ import print_function
from os import path as fp
import os
import sys

from joblib import (delayed)

from attelo.io import Torpor, write_predictions_output
from attelo.decoding.util import (prediction_to_triples)
from attelo.fold import (select_training,
                         select_testing)
from attelo.harness.util import (makedirs)


def _eval_banner(econf, hconf, fold):
    """
    Which combo of eval parameters are we running now?
    """
    msg = ("Reassembling "
           "fold {fnum} [{dset}]\t"
           "parser: {parser}")
    return msg.format(fnum=fold,
                      dset=hconf.dataset,
                      parser=econf.parser.key)


def _tmp_output_filename(path, suffix):
    """
    Temporary filename for output file segment
    """
    return fp.join(fp.dirname(path),
                   '_' + fp.basename(path) + '.' + suffix)


def concatenate_outputs(mpack, output_path):
    """
    (For use after :py:func:`delayed_main_for_harness`)

    Concatenate temporary per-group outputs into a single
    combined output
    """
    tmpfiles = [_tmp_output_filename(output_path, d)
                for d in sorted(mpack.keys())]
    with open(output_path, 'wb') as file_out:
        for tfile in tmpfiles:
            with open(tfile, 'rb') as file_in:
                file_out.write(file_in.read())
    for tmpfile in tmpfiles:
        os.remove(tmpfile)


def _parse_group(dpack, parser, output_path):
    '''
    parse a single group and write its output

    score the predictions if we have

    :rtype Count or None
    '''
    dpack = parser.transform(dpack)
    # we trust the parser to select what it thinks is its best prediction
    prediction = prediction_to_triples(dpack)
    write_predictions_output(dpack, prediction, output_path)


def jobs(mpack, parser, output_path):
    """Get a list of delayed decoding jobs for the documents in this group.

    Parameters
    ----------
    mpack : DataPack
        TODO
    parser : TODO
        TODO
    output_path : string
        Output path

    Returns
    -------
    res : list of delayed calls produced by joblib.delayed
    """
    # * clean temp files
    tmpfiles = [_tmp_output_filename(output_path, d)
                for d in mpack.keys()]
    for tmpfile in tmpfiles:
        if fp.exists(tmpfile):
            os.remove(tmpfile)
    # * generate delayed decoding jobs
    res = [delayed(_parse_group)(dpack, parser,
                                 _tmp_output_filename(output_path, onedoc))
           for onedoc, dpack in mpack.items()]
    return res


def learn(hconf, econf, dconf, fold):
    """
    Run the learners for the given configuration
    """
    if fold is None:
        subpacks = dconf.pack
        parent_dir = hconf.combined_dir_path()
    else:
        subpacks = select_training(dconf.pack, dconf.folds, fold)
        parent_dir = hconf.fold_dir_path(fold)

    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    cache = hconf.model_paths(econf.learner, fold, econf.parser)
    with Torpor('learning {}'.format(econf.key)):
        dpacks = subpacks.values()
        targets = [d.target for d in dpacks]
        econf.parser.payload.fit(dpacks, targets, cache=cache)


def delayed_decode(hconf, dconf, econf, fold):
    """
    Return possible futures for decoding groups within
    this model/decoder combo for the given fold
    """
    if fold is None and hconf.test_evaluation is None:
        return []
    if _say_if_decoded(hconf, econf, fold, stage='decoding'):
        return []

    output_path = hconf.decode_output_path(econf, fold)
    makedirs(fp.dirname(output_path))

    if fold is None:
        subpack = dconf.pack
    else:
        subpack = select_testing(dconf.pack, dconf.folds, fold)

    parser = econf.parser.payload

    res = jobs(subpack, parser, output_path)
    return res


def decode_on_the_fly(hconf, dconf, fold):
    """Learn each parser, returning decoder jobs as each is learned.

    Yields decoder jobs, which should hopefully allow us to effectively
    learn and decode in parallel.
    """
    for econf in hconf.evaluations:
        learn(hconf, econf, dconf, fold)
        for job in delayed_decode(hconf, dconf, econf, fold):
            yield job


def _say_if_decoded(hconf, econf, fold, stage='decoding'):
    """
    If we have already done the decoding for a given config
    and fold, say so and return True
    """
    if fp.exists(hconf.decode_output_path(econf, fold)):
        print(("skipping {stage} {parser} "
               "(already done)").format(stage=stage,
                                        parser=econf.parser.key),
              file=sys.stderr)
        return True
    else:
        return False


def post_decode(hconf, dconf, econf, fold):
    """
    Join together output files from this model/decoder combo
    """
    if _say_if_decoded(hconf, econf, fold, stage='reassembly'):
        return

    print(_eval_banner(econf, hconf, fold), file=sys.stderr)
    if fold is None:
        subpack = dconf.pack
    else:
        subpack = select_testing(dconf.pack, dconf.folds, fold)
    concatenate_outputs(subpack,
                        hconf.decode_output_path(econf, fold))
