"""
Saving and loading data or models
"""

from __future__ import print_function
from itertools import chain
import codecs
import copy
import csv
import json
import sys
import time
import traceback

from sklearn.datasets import load_svmlight_file

import educe  # WIP

from .cdu import CDU
from .edu import (EDU, FAKE_ROOT_ID, FAKE_ROOT)
from .table import (DataPack, DataPackException,
                    UNKNOWN, UNRELATED,
                    get_label_string, groupings)
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


def load_cdus(cdu_file):
    """Load the description of CDUs.

    As of 2016-07-28, this is WIP.

    Parameters
    ----------
    cdu_file: pathname
        Path to a file that describes CDUs. Each line provides the
        identifier of the CDU, then the list of its member DUs.

    Returns
    -------
    
    """
    with open(cdu_file, 'rb') as instream:
        reader = csv.reader(instream, dialect=csv.excel_tab)
        return [CDU(x[0], tuple(x[1:])) for x in reader if x]


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


def _process_cdu_links(cdus, edus, pairings):
    """Convert from the results of `load_cdus` and `load_pairings` to a
    sequence of CDUs and pairings respectively.

    Parameters
    ----------
    cdus: list of CDU
        CDUs
    edus: list of EDU
        EDUs
    pairings: list of pairs of string
        List of pairings from/to CDUs

    Returns
    -------
    cdus: list of CDU
        CDUs
    pairings: list of pairs of (EDU or CDU, EDU or CDU)
        List of pairs of DUs
    """
    du_map = {e.id: e for e in edus}
    du_map.update({x.id: x for x in cdus})

    du_names = frozenset(chain.from_iterable(pairings))

    naughty = [x for x in du_names if x not in du_map]
    if naughty:
        oops = ('The pairings file mentions the following DUs but the EDU '
                'and CDU files do not actually include DUs to go with them:'
                ' {}')
        raise DataPackException(oops.format(truncate(', '.join(naughty),
                                                     1000)))
    pairings = [(du_map[src], du_map[tgt]) for src, tgt in pairings]

    return cdus, pairings
    


def load_multipack(edu_file, pairings_file, feature_file, vocab_file,
                   cdu_file=None, cdu_pairings_file=None,
                   cdu_feature_file=None,
                   corpus_path=None,  # WIP
                   verbose=False):
    """Read EDUs and features for edu pairs.

    Perform some basic sanity checks, raising
    :py:class:`IoException` if they should fail

    Parameters
    ----------
    ... TODO

    corpus_path : string
        Path to the labelled corpus, to retrieve the original gold
        structures ; at the moment, only works with the RST corpus to
        access gold RST constituency trees.

    Returns
    -------
    mpack: Multipack
        Multipack (= dict) from grouping to DataPack.
    """
    vocab = load_vocab(vocab_file)

    with Torpor("Reading edus and pairings", quiet=not verbose):
        edus, pairings = _process_edu_links(load_edus(edu_file),
                                            load_pairings(pairings_file))

    with Torpor("Reading features", quiet=not verbose):
        labels = [UNKNOWN] + load_labels(feature_file)
        # pylint: disable=unbalanced-tuple-unpacking
        data, targets = load_svmlight_file(feature_file,
                                           n_features=len(vocab))
        # pylint: enable=unbalanced-tuple-unpacking

    # WIP
    if (cdu_file is not None
        and cdu_pairings_file is not None
        and cdu_feature_file is not None):
        # augment DataPack with CDUs and pairings from/to them
        with Torpor("Reading CDUs and pairings", quiet=not verbose):
            cdus, cdu_pairings = _process_cdu_links(
                load_cdus(cdu_file),
                edus,
                load_pairings(cdu_pairings_file))

        with Torpor("Reading features", quiet=not verbose):
            # CDU files use the same label set as the EDU files ; this is
            # not really enforced, but it is implemented this way in
            # irit_rst_dt.cmd.gather
            # pylint: disable=unbalanced-tuple-unpacking
            cdu_data, cdu_targets = load_svmlight_file(
                cdu_feature_file, n_features=len(vocab))
            # pylint: enable=unbalanced-tuple-unpacking
    else:
        cdus = None
        cdu_pairings = None
        cdu_data = None
        cdu_targets = None

    # WIP augment DataPack with the gold structure for each grouping
    if corpus_path is None:
        ctargets = {}
    else:
        corpus_reader = educe.rst_dt.corpus.Reader(corpus_path)
        # FIXME should be [v] so that it is adapted to forests (lists)
        # of structures, e.g. produced by for_intra()
        ctargets = {k.doc: v for k, v in corpus_reader.slurp().items()}
        # TODO modify educe.rst_dt.corpus.Reader.slurp_subcorpus() to
        # convert fine-grained to coarse-grained relations by default,
        # e.g. add kwarg coarse_rels=True, then find all current callers
        # but this one and call slurp* with coarse_rels=False
    # end WIP

    with Torpor("Build data packs", quiet=not verbose):
        dpack = DataPack.load(edus, pairings, data, targets, ctargets,
                              cdus, cdu_pairings, cdu_data, cdu_targets,
                              labels, vocab)

    mpack = {grp_name: dpack.selected(idxs)
             for grp_name, idxs in groupings(pairings).items()}
    return mpack


def load_vocab(filename):
    """Read feature vocabulary"""
    features = []
    with codecs.open(filename, 'r', 'utf-8') as stream:
        for line in stream:
            features.append(line.split('\t')[0])
    return features

# ---------------------------------------------------------------------
# predictions
# ---------------------------------------------------------------------


def write_predictions_output(dpack, predicted, filename):
    """
    Write predictions to an output file whose format
    is documented in :doc:`../output`
    """
    links = {}
    for edu1, edu2, label in predicted:
        links[(edu1, edu2)] = label

    def mk_row(edu1, edu2):
        'return a list of columns'
        edu1_id = edu1.id
        edu2_id = edu2.id
        row = [edu1_id,
               edu2_id,
               links.get((edu1_id, edu2_id), UNRELATED)]
        return [x.encode('utf-8') for x in row]

    with open(filename, 'wb') as fout:
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
        return tuple(x.decode('utf-8') for x in row)

    with open(edu_file, 'rb') as instream:
        reader = csv.reader(instream, dialect=csv.excel_tab)
        return [mk_pair(r) for r in reader if r]


def load_gold_predictions(pairings_file, feature_file, verbose=False):
    """
    Load a pairings and feature file as though it were a set of
    predictions

    :rtype: [(string, string, string)]
    """
    pairings = load_pairings(pairings_file)
    with Torpor("Reading features", quiet=not verbose):
        labels = load_labels(feature_file)
        # pylint: disable=unbalanced-tuple-unpacking
        _, targets = load_svmlight_file(feature_file)
        # pylint: enable=unbalanced-tuple-unpacking
    return [(x1, x2, get_label_string(labels, t))
            for ((x1, x2), t) in zip(pairings, targets)]


# ---------------------------------------------------------------------
# folds
# ---------------------------------------------------------------------
def load_fold_dict(filename):
    """
    Load fold dictionary into memory from file
    """
    with open(filename, 'r') as stream:
        return json.load(stream)


def save_fold_dict(fold_dict, filename):
    """
    Dump fold dictionary to a file
    """
    with open(filename, 'w') as stream:
        json.dump(fold_dict, stream, indent=2)
