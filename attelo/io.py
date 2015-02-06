"""
Saving and loading data or models
"""

from __future__ import print_function
from collections import defaultdict
from itertools import chain
from os import path as fp
import codecs
import copy
import csv
import os
import sys
import time
import traceback

import joblib
from sklearn.datasets import load_svmlight_file

from .edu import (EDU, FAKE_ROOT_ID, FAKE_ROOT)
from .table import (DataPack, DataPackException, UNRELATED)
from .util import truncate

# pylint: disable=too-few-public-methods


class IoException(Exception):
    """
    Exceptions related to reading/writing data
    """
    def __init__(self, msg):
        super(IoException, self).__init__(msg)

# ---------------------------------------------------------------------
# feedback
# ---------------------------------------------------------------------


# pylint: disable=redefined-builtin, invalid-name
class Torpor(object):
    """
    Announce that we're about to do something, then do it,
    then say we're done.

    Usage: ::

        with Torpor("doing a slow thing"):
            some_slow_thing

    Output (1): ::

        doing a slow thing...

    Output (2a): ::

        doing a slow thing... done

    Output (2b): ::

        doing a slow thing... ERROR
        <stack trace>

    :param quiet: True to skip the message altogether
    """
    def __init__(self, msg,
                 sameline=True,
                 quiet=False,
                 file=sys.stderr):
        self._msg = msg
        self._file = file
        self._sameline = sameline
        self._quiet = quiet
        self._start = 0
        self._end = 0

    def __enter__(self):
        # we grab the wall time instead of using time.clock() (A)
        # because we # are not using this for profiling but just to
        # get a rough idea what's going on, and (B) because we want
        # to include things like IO into the mix
        self._start = time.time()
        if self._quiet:
            return
        elif self._sameline:
            print(self._msg, end="... ", file=self._file)
        else:
            print("[start]", self._msg, file=self._file)

    def __exit__(self, type, value, tb):
        self._end = time.time()
        if tb is None:
            if not self._quiet:
                done = "done" if self._sameline else "[-end-] " + self._msg
                ms_elapsed = 1000 * (self._end - self._start)
                final_msg = u"{} [{:.0f} ms]".format(done, ms_elapsed)
                print(final_msg, file=self._file)
        else:
            if not self._quiet:
                oops = "ERROR!" if self._sameline else "ERROR! " + self._msg
                print(oops, file=self._file)
            traceback.print_exception(type, value, tb)
            sys.exit(1)
# pylint: redefined-builtin, invalid-name


# ---------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------


def load_edus(edu_file):
    """
    Read EDUs (see :doc:`../input`)

    :rtype: [EDU]

    .. _format: https://github.com/kowey/attelo/doc/inputs.rst
    """
    def read_edu(row):
        'interpret a single row'
        expected_len = 6
        if len(row) != expected_len:
            oops = ('This row in the EDU file {efile} has {num} '
                    'elements instead of the expected {expected}: '
                    '{row}')
            raise IoException(oops.format(efile=edu_file,
                                          num=len(row),
                                          expected=expected_len,
                                          row=row))
        [global_id, txt, grouping, subgrouping, start_str, end_str] = row
        start = int(start_str)
        end = int(end_str)
        return EDU(global_id,
                   txt.decode('utf-8'),
                   start,
                   end,
                   grouping,
                   subgrouping)

    with open(edu_file, 'rb') as instream:
        reader = csv.reader(instream, dialect=csv.excel_tab)
        return [read_edu(r) for r in reader if r]


def load_pairings(edu_file):
    """
    Read and return EDU pairings (see :doc:`../input`).
    We assume the order is parent, child

    :rtype: [(string, string)]

    .. _format: https://github.com/kowey/attelo/doc/inputs.rst
    """
    def read_pair(row):
        'interpret a single row'
        if len(row) < 2 or len(row) > 3:
            oops = ('This row in the pairings file {efile} has '
                    '{num} elements instead of the expected 2 or 3')
            raise IoException(oops.format(efile=edu_file,
                                          num=len(row),
                                          row=row))
        return tuple(row[:2])

    with open(edu_file, 'rb') as instream:
        reader = csv.reader(instream, dialect=csv.excel_tab)
        return [read_pair(r) for r in reader if r]


def load_labels(feature_file):
    """
    Read the very top of a feature file and read the labels comment,
    return the sequence of labels, else return None

    :rtype: [string] or None
    """
    with codecs.open(feature_file, 'r', 'utf-8') as stream:
        line = stream.readline()
        if line.startswith('#'):
            seq = line[1:].split()
            if seq[0] == 'labels:':
                return seq[1:]
    # fall-through case, no labels found
    return None


def _process_edu_links(edus, pairings):
    """
    Convert from the results of :py:method:load_edus: and
    :py:method:load_pairings: to a sequence of edus and pairings
    respectively

    :rtype: ([EDU], [(EDU,EDU)])
    """
    edumap = {e.id: e for e in edus}
    enames = frozenset(chain.from_iterable(pairings))

    if FAKE_ROOT_ID in enames:
        edus2 = [FAKE_ROOT] + edus
        edumap[FAKE_ROOT_ID] = FAKE_ROOT
    else:
        edus2 = copy.copy(edus)

    naughty = [x for x in enames if x not in edumap]
    if naughty:
        oops = ('The pairings file mentions the following EDUs but the EDU '
                'file does not actually include EDUs to go with them: {}')
        raise DataPackException(oops.format(truncate(', '.join(naughty),
                                                     1000)))

    pairings2 = [(edumap[e1], edumap[e2]) for e1, e2 in pairings]
    return edus2, pairings2


def load_data_pack(edu_file, pairings_file, feature_file, verbose=False):
    """
    Read EDUs and features for edu pairs.

    Perform some basic sanity checks, raising
    :py:class:`IoException` if they should fail

    :rtype: :py:class:`DataPack` or None
    """
    with Torpor("Reading edus and pairings", quiet=not verbose):
        edus, pairings = _process_edu_links(load_edus(edu_file),
                                            load_pairings(pairings_file))

    with Torpor("Reading features", quiet=not verbose):
        labels = load_labels(feature_file)
        # pylint: disable=unbalanced-tuple-unpacking
        data, targets = load_svmlight_file(feature_file)
        # pylint: enable=unbalanced-tuple-unpacking

    with Torpor("Sanity checking data pack", quiet=not verbose):
        dpack = DataPack.load(edus, pairings, data, targets, labels)

    return dpack

# ---------------------------------------------------------------------
# predictions
# ---------------------------------------------------------------------


def start_predictions_output(filename):
    """
    Initialise any output files that are to be appended to rather
    than written separately
    """
    dname = fp.dirname(filename)
    if not fp.exists(dname):
        os.makedirs(dname)
    open(filename, 'wb').close()


def append_predictions_output(dpack, predicted, filename):
    """
    Append the predictions to a CONLL like output file documented in
    :doc:`../output`

    See also :py:func:`start_predictions_output`
    """
    links = {}
    for edu1, edu2, label in predicted:
        links[(edu1, edu2)] = label

    def mk_row(edu1, edu2):
        'return a list of columns'
        edu1_id = edu1.id
        edu2_id = edu2.id
        return [edu1_id,
                edu2_id,
                links.get((edu1_id, edu2_id), UNRELATED)]

    with open(filename, 'a') as fout:
        writer = csv.writer(fout, dialect=csv.excel_tab)
        # by convention the zeroth edu is the root node
        for edu1, edu2 in dpack.pairings:
            writer.writerow(mk_row(edu1, edu2))


def load_predictions(edu_file):
    """
    Read back predictions (see :doc:`../output`), returning a list
    of triples: parent id, child id, relation label (or 'UNRELATED')

    :rtype: [(string, string, string)]
    """
    def mk_pair(row):
        'interpret a single row'
        expected_len = 3
        if len(row) < expected_len:
            oops = ('This row in the predictions file {efile} has {num} '
                    'elements instead of the expected {expected}: '
                    '{row}')
            raise IoException(oops.format(efile=edu_file,
                                          num=len(row),
                                          expected=expected_len,
                                          row=row))
        return tuple(row)

    with open(edu_file, 'rb') as instream:
        reader = csv.reader(instream, dialect=csv.excel_tab)
        return [mk_pair(r) for r in reader if r]


# ---------------------------------------------------------------------
# models
# ---------------------------------------------------------------------


def load_model(filename):
    """
    Load model into memory from file

    :rtype: sklearn classifier
    """
    return joblib.load(filename)


def save_model(filename, model):
    """
    Dump model into a file

    :type: model: sklearn classifier
    """
    joblib.dump(model, filename)
