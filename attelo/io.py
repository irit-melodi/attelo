"""
Saving and loading data or models
"""

from __future__ import print_function
from itertools import chain
import codecs
import copy
import csv
from glob import glob
import json
import os
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
    cdu_file : pathname
        Path to a file that describes CDUs. Each line provides the
        identifier of the CDU, then the list of its member DUs.

    Returns
    -------
    cdus : list of CDU
        CDUs built from the file content.
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


def load_labels(labels_file):
    """Read the list of labels.

    Returns
    -------
    labels: list of strings
        List of labels
    """
    lbl_map = dict()
    with codecs.open(labels_file, 'r', 'utf-8') as stream:
        for line in stream:
            i, lbl = line.strip().split()
            lbl_map[lbl] = int(i)
    labels = [lbl for lbl, i in sorted(lbl_map.items(), key=lambda x: x[1])]
    return labels


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
    cdus : list of CDU
        CDUs
    edus : list of EDU
        EDUs
    pairings : list of pairs of string
        List of pairings from/to CDUs

    Returns
    -------
    cdus : list of CDU
        CDUs
    pairings : list of pairs of (EDU or CDU, EDU or CDU)
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


def _load_multipack_cdus(cdu_file, cdu_pairings_file, cdu_feature_file,
                         vocab, doc_names):
    """Helper to load the CDU part of a multipack.

    Parameters
    ----------
    cdu_file : str
        CDU file

    cdu_pairings_file : str
        CDU pairings file

    cdu_feature_file : str
        Feature file for CDU pairings

    vocab : dict(str, int)?
        Feature vocabulary

    doc_names : :obj:`list` of str
        List of document names.

    Returns
    -------
    doc_cdus : dict(str, TODO)
        Map document names to CDUs

    doc_cdu_pairings : dict(str, TODO)
        Map document names to CDU pairings

    doc_cdu_data : dict(str, TODO)
        Map document names to CDU data (features).

    doc_cdu_targets : dict(str, TODO)
        Map document names to CDU targets.
    """
    if (cdu_file is not None
        and cdu_pairings_file is not None
        and cdu_feature_file is not None):
        # WIP one file per doc
        cdu_files = {os.path.basename(f).rsplit('.', 4)[0]: f
                     for f in glob(cdu_file)}
        cdu_pairings_files = {os.path.basename(f).rsplit('.', 4)[0]: f
                              for f in glob(cdu_pairings_file)}
        cdu_feature_files = {os.path.basename(f).rsplit('.', 3)[0]: f
                             for f in glob(cdu_feature_file)}
    else:
        cdu_files = None
        cdu_pairings_files = None
        cdu_feature_files = None

    doc_cdus = dict()
    doc_cdu_pairings = dict()
    doc_cdu_data = dict()
    doc_cdu_targets = dict()
    if (cdu_files is not None
        and cdu_pairings_files is not None
        and cdu_feature_files is not None):
        # augment DataPack with CDUs and pairings from/to them
        with Torpor("Reading CDUs and pairings", quiet=not verbose):
            for doc_name in doc_names:
                # one file per doc
                cdu_f = cdu_files[doc_name]
                cdu_pairings_f = cdu_pairings_files[doc_name]
                # end one file per doc
                cdus, cdu_pairings = _process_cdu_links(
                    load_cdus(cdu_f),
                    edus,
                    load_pairings(cdu_pairings_f))
                doc_cdus[doc_name] = cdus
                doc_cdu_pairings[doc_name] = cdu_pairings

        with Torpor("Reading features for CDU pairings", quiet=not verbose):
            for doc_name in doc_names:
                cdu_feature_f = cdu_feature_files[doc_name]
                if doc_cdu_pairings[doc_name]:
                    # CDU files use the same label set as the EDU files ;
                    # this is not really enforced, but it is implemented
                    # this way in irit_rst_dt.cmd.gather
                    # pylint: disable=unbalanced-tuple-unpacking
                    cdu_data, cdu_targets = load_svmlight_file(
                        cdu_feature_f, n_features=len(vocab))
                else:
                    cdu_data = None
                    cdu_targets = None
                # pylint: enable=unbalanced-tuple-unpacking
                doc_cdu_data[doc_name] = cdu_data
                doc_cdu_targets[doc_name] = cdu_targets
    else:
        cdus = None
        cdu_pairings = None
        cdu_data = None
        cdu_targets = None
        # build dictionaries for CDU stuff
        doc_cdus = {doc_name: None for doc_name in doc_names}
        doc_cdu_pairings = {doc_name: None for doc_name in doc_names}
        doc_cdu_data = {doc_name: None for doc_name in doc_names}
        doc_cdu_targets = {doc_name: None for doc_name in doc_names}
    # end WIP CDUs
    return tuple(doc_cdus, doc_cdu_pairings, doc_cdu_data, doc_cdu_targets)


def load_multipack(edu_file, pairings_file, feature_file, vocab_file,
                   labels_file,
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
    mpack = dict()

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

    # one file per doc (2016-08-30)
    edu_files = {os.path.basename(f).rsplit('.', 4)[0]: f
                 for f in glob(edu_file)}
    pairings_files = {os.path.basename(f).rsplit('.', 4)[0]: f
                      for f in glob(pairings_file)}
    feature_files = {os.path.basename(f).rsplit('.', 3)[0]: f
                     for f in glob(feature_file)}
    # end one file per doc

    # common files: vocabulary, labels
    vocab = load_vocab(vocab_file)
    labels = load_labels(labels_file)
    assert labels[0] == UNKNOWN

    doc_names = sorted(edu_files.keys())

    doc_edus = dict()
    doc_pairings = dict()
    with Torpor("Reading EDUs and pairings", quiet=not verbose):
        for doc_name in doc_names:
            edu_f = edu_files[doc_name]
            pairings_f = pairings_files[doc_name]
            edus, pairings = _process_edu_links(load_edus(edu_f),
                                                load_pairings(pairings_f))
            # each file should contain info from exactly one doc (grouping)
            grp_names = groupings(pairings).keys()
            assert grp_names == [doc_name]
            # store
            doc_edus[doc_name] = edus
            doc_pairings[doc_name] = pairings

    doc_data = dict()
    doc_targets = dict()
    with Torpor("Reading features", quiet=not verbose):
        for doc_name in doc_names:
            feature_f = feature_files[doc_name]
            # pylint: disable=unbalanced-tuple-unpacking
            data, targets = load_svmlight_file(feature_f,
                                               n_features=len(vocab))
            # pylint: enable=unbalanced-tuple-unpacking
            doc_data[doc_name] = data
            doc_targets[doc_name] = targets

    # WIP CDUs
    doc_cdus, doc_cdu_pairings, doc_cdu_data, doc_cdu_targets = _load_multipack_cdus(cdu_file, cdu_pairings_file, cdu_feature_file, vocab, doc_names)

    # build DataPack
    with Torpor("Build data packs", quiet=not verbose):
        for doc_name in doc_names:
            dpack = DataPack.load(
                doc_edus[doc_name], doc_pairings[doc_name],
                doc_data[doc_name], doc_targets[doc_name],
                # maybe we could avoid the dummy dict (that contains a
                # unique pair) and just have an RSTTree here, but I am
                # afraid it could cause inconsistencies with datapacks
                # created with Datapack.vstack
                dict([(doc_name, ctargets.get(doc_name, None))]),
                # WIP CDU
                doc_cdus[doc_name], doc_cdu_pairings[doc_name],
                doc_cdu_data[doc_name], doc_cdu_targets[doc_name],
                # end WIP CDU
                labels, vocab)
            mpack[doc_name] = dpack

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


def load_predictions(edges_file):
    """Read back predictions (see :doc:`../output`), returning a list
    of triples: parent id, child id, relation label (or 'UNRELATED').

    Parameters
    ----------
    edges_file: str
        Path to the file that contains predicted edges.

    Returns
    -------
    edges_pred: list of (str, str, str)
        List of predicted edges as triples (gov_id, dep_id, lbl_pred).
        If (gov_id, dep_id) is predicted to be unattached, lbl_pred is
        'UNRELATED'.
    """
    def mk_pair(row):
        'interpret a single row'
        expected_len = 3
        if len(row) < expected_len:
            oops = ('This row in the predictions file {efile} has {num} '
                    'elements instead of the expected {expected}: '
                    '{row}')
            raise IoException(oops.format(efile=edges_file,
                                          num=len(row),
                                          expected=expected_len,
                                          row=row))
        return tuple(x.decode('utf-8') for x in row)

    with open(edges_file, 'rb') as instream:
        reader = csv.reader(instream, dialect=csv.excel_tab)
        return [mk_pair(r) for r in reader if r]


def load_gold_predictions(pairings_file, feature_file, labels_file,
                          verbose=False):
    """
    Load a pairings and feature file as though it were a set of
    predictions

    :rtype: [(string, string, string)]
    """
    labels = load_labels(labels_file)

    pairings = load_pairings(pairings_file)
    with Torpor("Reading features", quiet=not verbose):
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
