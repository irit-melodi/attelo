'''
Control over attelo parsers as might be needed for a test harness
'''

from __future__ import print_function
from os import path as fp
import os

from joblib import (delayed)

from ..io import (write_predictions_output)
from attelo.decoding.util import (prediction_to_triples)


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
    """
    Return a list of delayed decoding jobs for the various
    documents in this group
    """
    res = []
    tmpfiles = [_tmp_output_filename(output_path, d)
                for d in mpack.keys()]
    for tmpfile in tmpfiles:
        if fp.exists(tmpfile):
            os.remove(tmpfile)
    for onedoc, dpack in mpack.items():
        tmp_output_path = _tmp_output_filename(output_path, onedoc)
        res.append(delayed(_parse_group)(dpack, parser, tmp_output_path))
    return res
