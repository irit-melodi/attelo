"""
Saving and loading data or models
"""

from __future__ import print_function
from collections import defaultdict
from itertools import chain
from os import path as fp
import csv
import os
import sys
import time
import traceback

import joblib
from sklearn.datasets import load_svmlight_file

from .edu import (EDU, FAKE_ROOT_ID, FAKE_ROOT)
from .table import DataPack, DataPackException

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
    Read EDUs (see :ref:`edu-input-format`), returning a list
    of EDUs paired with ids for their possible parents.

    Note that the order that the EDUs and parents are returned
    in is significant as it is used to for indexing the feature
    file

    :rtype [(EDU, [String])]

    .. _format: https://github.com/kowey/attelo/doc/inputs.rst
    """
    def mk_pair(row):
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
        [global_id, txt, grouping, start_str, end_str, parents_str] = row
        start = int(start_str)
        end = int(end_str)
        edu = EDU(global_id,
                  txt.decode('utf-8'),
                  start,
                  end,
                  grouping)
        parents = parents_str.split()
        return edu, parents

    with open(edu_file, 'rb') as instream:
        reader = csv.reader(instream, dialect=csv.excel_tab)
        return [mk_pair(r) for r in reader if r]


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
    :ref:output-format:

    See also :py:method:start_predictions_file:
    """
    incoming = defaultdict(list)
    for edu1, edu2, label in predicted:
        incoming[edu2].append((edu1, label))
    max_indegree = max(len(x) for x in incoming.items()) if incoming else 1

    def mk_row(edu):
        "csv row for the given edu"

        parents = incoming.get(edu.id)
        if parents:
            linkstuff = list(chain.from_iterable(parents))
        else:
            linkstuff = ["0", "ROOT"]
        pad_len = max_indegree * 2 - len(linkstuff)
        padding = [''] * pad_len
        return [edu.id,
                edu.text.encode('utf-8'),
                edu.grouping,
                edu.start,
                edu.end] + linkstuff + padding

    with open(filename, 'a') as fout:
        writer = csv.writer(fout, dialect=csv.excel_tab)
        for edu in dpack.edus:
            writer.writerow(mk_row(edu))


def load_data_pack(edu_file, feature_file, verbose=False):
    """
    Read EDUs and features for edu pairs.

    Perform some basic sanity checks, raising
    :py:class:IoException: if they should fail

    :rtype :py:class:DataPack: or None
    """
    with Torpor("Reading edus", quiet=not verbose):
        edulinks = load_edus(edu_file)

    edumap = {e.id: e for e, _ in edulinks}
    parents = list(chain.from_iterable(l for _, l in edulinks))

    if FAKE_ROOT_ID in parents:
        edus = [FAKE_ROOT]
        edumap[FAKE_ROOT_ID] = FAKE_ROOT
    else:
        edus = []

    # this is not quite the same as the DataPack._check_edu_pairings()
    # because here are working only with identifiers and not objects
    naughty = [x for x in parents if x not in edumap]
    if naughty:
        oops = ('The EDU files mentions the following candidate parent ids, '
                'but does not actually include EDUs to go with them: {}')
        raise DataPackException(oops.format(', '.join(naughty)))

    pairings = []
    for edu, links in edulinks:
        edus.append(edu)
        pairings.extend((edu, edumap[l]) for l in links)

    with Torpor("Reading features", quiet=not verbose):
        data, targets = load_svmlight_file(feature_file)

    return DataPack.load(edus, pairings, data, targets)


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
