'''
Control over attelo decoders as might be needed for a test harness
'''

from __future__ import print_function
from os import path as fp
import os

from joblib import (delayed)

from ..io import (write_predictions_output)
from ..decoding import (DecoderException, decode)


def _tmp_output_filename(path, suffix):
    """
    Temporary filename for output file segment
    """
    return fp.join(fp.dirname(path),
                   '_' + fp.basename(path) + '.' + suffix)


def concatenate_outputs(dpack, output_path):
    """
    (For use after :py:func:`delayed_main_for_harness`)

    Concatenate temporary per-group outputs into a single
    combined output
    """
    groupings = dpack.groupings()
    tmpfiles = [_tmp_output_filename(output_path, d)
                for d in groupings]
    with open(output_path, 'wb') as file_out:
        for tfile in tmpfiles:
            with open(tfile, 'rb') as file_in:
                file_out.write(file_in.read())
    for tmpfile in tmpfiles:
        os.remove(tmpfile)


def _decode_group(dpack, models, decoder, mode,
                  output_path):
    '''
    decode a single group and write its output

    score the predictions if we have

    :rtype Count or None
    '''
    predictions = decode(dpack, models, decoder, mode)
    if not predictions:
        raise DecoderException('decoder must make at least one prediction')

    # we trust the decoder to select what it thinks is its best prediction
    first_prediction = predictions[0]
    write_predictions_output(dpack, first_prediction, output_path)


def jobs(dpack, models, decoder, mode, output_path):
    """
    Return a list of delayed decoding jobs for the various
    documents in this group
    """
    groupings = dpack.groupings()
    res = []
    tmpfiles = [_tmp_output_filename(output_path, d)
                for d in groupings]
    for tmpfile in tmpfiles:
        if fp.exists(tmpfile):
            os.remove(tmpfile)
    for onedoc, indices in groupings.items():
        onepack = dpack.selected(indices)
        tmp_output_path = _tmp_output_filename(output_path, onedoc)
        res.append(delayed(_decode_group)(onepack, models, decoder, mode,
                                          tmp_output_path))
    return res
