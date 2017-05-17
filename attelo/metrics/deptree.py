"""Common metrics on dependency trees.
"""

from __future__ import absolute_import, print_function

from collections import Counter
import itertools

import numpy as np


def compute_uas_las(dtree_true, dtree_pred, metrics=None, doc_names=None):
    """Compute dependency metrics for trees in dtree_pred wrt dtree_true.

    The computed metrics are the traditional UAS and LAS, plus LS
    for Labelling Score (counts of correct labels, regardless of head).

    Parameters
    ----------
    dtree_true : list of RstDepTree
        Reference trees

    dtree_pred : list of RstDepTree
        Predicted trees

    metrics : list of str
        If None, defaults to ['U', 'R'] aka. UAS and LAS. Possible
        values in {'U', 'R', 'R+N', 'R+O', 'F', 'tag_R'}.

    Returns
    -------
    res : tuple of float
        Score for each metric in order.
    """
    # 'U': correct unlabelled deps ; was: nb_ua_ok
    # 'R': correct labelled deps ; was: nb_la_ok
    # 'tag_R': correct labellings: right labels, possibly wrong heads ; was: nb_l_ok
    # 'R+N': relation and nuclearity
    # 'R+O': relation and order
    # 'F': relation, order, nuclearity
    nb_tp = Counter({k: 0 for k in metrics})
    nb_tot = 0  # total deps

    for i, (dt_true, dt_pred) in enumerate(
            itertools.izip(dtree_true, dtree_pred)):
        if doc_names is not None:
            doc_name = doc_names[i]  # for verbose/debug
        tp = dict()
        tp_bins = dict()
        # exclude fake root from metrics
        # head : dependencies
        if any(x in set(['U', 'N', 'R', 'O', 'R+N', 'R+O', 'F'])
               for x in metrics):
            heads_true = np.array(dt_true.heads[1:])
            heads_pred = np.array(dt_pred.heads[1:])
            tp['U'] = heads_true == heads_pred
            tp_bins['U'] = heads_true[tp['U']]
        # relation tag
        if any(x in set(['tag_R', 'R', 'R+N', 'R+O', 'F']) for x in metrics):
            labels_true = np.array(dt_true.labels[1:])
            labels_pred = np.array(dt_pred.labels[1:])
            tp['tag_R'] = labels_true == labels_pred
            tp_bins['tag_R'] = labels_true[tp['tag_R']]
            # dep
            tp['R'] = np.logical_and(tp['U'], tp['tag_R'])
            tp_bins['R'] = labels_true[tp['R']]
        # nuclearity tag
        if any(x in set(['tag_N', 'N', 'R+N', 'F']) for x in metrics):
            nucs_true = np.array(dt_true.nucs[1:])
            nucs_pred = np.array(dt_pred.nucs[1:])
            tp['tag_N'] = nucs_true == nucs_pred
            tp_bins['tag_N'] = nucs_true[tp['tag_N']]
            # dep
            tp['N'] = np.logical_and(tp['U'], tp['tag_N'])
            tp_bins['N'] = nucs_true[tp['N']]
        # order tag
        if any(x in set(['tag_O', 'O', 'R+O', 'F']) for x in metrics):
            rnks_true = np.array(dt_true.ranks[1:])
            rnks_pred = np.array(dt_pred.ranks[1:])
            tp['tag_O'] = rnks_true == rnks_pred
            tp_bins['tag_O'] = rnks_true[tp['tag_O']]
            # dep
            tp['O'] = np.logical_and(tp['U'], tp['tag_O'])
            tp_bins['O'] = rnks_true[tp['O']]

        # dep on complex labels, build on simpler labelled deps
        if 'R+O' in metrics:
            tp['R+O'] = np.logical_and(tp['R'], tp['O'])
            tp_bins['R+O'] = np.array([
                (x, y) for x, y in zip(
                    labels_true[tp['R+O']], rnks_true[tp['R+O']])
            ])
        if 'R+N' in metrics:
            tp['R+N'] = np.logical_and(tp['R'], tp['N'])
            tp_bins['R+N'] = np.array([
                (x, y) for x, y in zip(
                    labels_true[tp['R+N']], nucs_true[tp['R+N']])
            ])
        if 'F' in metrics:
            tp['F'] = np.logical_and.reduce((tp['R'], tp['N'], tp['O']))
            tp_bins['F'] = np.array([
                (x, y, z) for x, y, z in zip(
                    labels_true[tp['F']], nucs_true[tp['F']],
                    rnks_true[tp['F']])
            ])

        # for each metric, update the number of true positives with
        # the count for the current instance
        for k, v in nb_tp.items():
            nb_tp[k] = v + len(tp_bins[k])

        nb_tot += len(heads_true)

    scores = {k: float(nb_tp[k]) / nb_tot for k in nb_tp}
    res = tuple([scores[k] for k in metrics])
    return res


def compute_uas_las_listcomp(dtree_true, dtree_pred):
    """Compute dependency metrics for trees in dtree_pred wrt dtree_true.

    The computed metrics are the traditional UAS and LAS, plus LS
    for Labelling Score (counts of correct labels, regardless of head).

    Alternative implementation that uses list comprehensions.

    Parameters
    ----------
    dtree_true: list of RstDepTree
        Reference trees

    dtree_pred: list of RstDepTree
        Predicted trees

    Returns
    -------
    (uas, las, ls): (float, float, float)
        The Unlabelled and Labelled Attachment Scores, plus the
        Labelling Score (new).
    """
    nb_ua_ok = 0  # correct unlabelled deps
    nb_la_ok = 0  # correct labelled deps
    nb_l_ok = 0  # correct labellings (right labels, possibly wrong heads)
    nb_tot = 0  # total deps
    for dt_true, dt_pred in itertools.izip(dtree_true, dtree_pred):
        # heads and labels are stored as two lists
        # exclude fake root from metrics
        heads_true = dt_true.heads[1:]
        labels_true = dt_true.labels[1:]

        heads_pred = dt_pred.heads[1:]
        labels_pred = dt_pred.labels[1:]

        # list comprehensions to do pseudo-vectorized operations
        gov_ok = [heads_pred[i] == heads_true[i]
                  for i in range(len(heads_pred))]
        lbl_ok = [labels_pred[i] == labels_true[i]
                  for i in range(len(labels_pred))]
        gov_lbl_ok = [gov_ok[i] and lbl_ok[i]
                      for i in range(len(gov_ok))]
        # update counts
        nb_ua_ok += sum(gov_ok)
        nb_la_ok += sum(gov_lbl_ok)
        nb_l_ok += sum(lbl_ok)
        nb_tot += len(heads_pred)

    score_uas = float(nb_ua_ok) / nb_tot
    score_las = float(nb_la_ok) / nb_tot
    score_ls = float(nb_l_ok) / nb_tot  # NEW

    return (score_uas, score_las, score_ls)


def compute_uas_las_np(dtree_true, dtree_pred):
    """Compute dependency metrics for trees in dtree_pred wrt dtree_true.

    The computed metrics are the traditional UAS and LAS, plus LS
    for Labelling Score (counts of correct labels, regardless of head).

    Alternative implementation that uses numpy.

    Parameters
    ----------
    dtree_true: list of RstDepTree
        Reference trees

    dtree_pred: list of RstDepTree
        Predicted trees

    Returns
    -------
    (uas, las, ls): (float, float, float)
        The Unlabelled and Labelled Attachment Scores, plus the
        Labelling Score (new).
    """
    uas_num = 0  # correct unlabelled deps
    las_num = 0  # correct labelled deps
    ls_num = 0  # correct labellings (right labels, possibly wrong heads)
    nb_tot = 0  # total deps
    for dt_true, dt_pred in itertools.izip(dtree_true, dtree_pred):
        # heads and labels are stored as two lists
        # exclude fake root from metrics
        heads_true = dt_true.heads[1:]
        labels_true = dt_true.labels[1:]

        heads_pred = dt_pred.heads[1:]
        labels_pred = dt_pred.labels[1:]

        # use numpy's truly vectorized operations:
        gov_ok = np.equal(heads_true, heads_pred)
        # element-wise comparison of arrays of strings is properly defined
        # only with the infix operator "=="
        # TODO change type of labels_* to arrays of ints, and use
        # np.equal(labels_true, labels_pred)
        lbl_ok = np.array(labels_true) == np.array(labels_pred)
        #
        uas_num += np.count_nonzero(gov_ok)
        las_num += np.count_nonzero(np.logical_and(gov_ok, lbl_ok))
        ls_num += np.count_nonzero(lbl_ok)
        nb_tot += gov_ok.size

    score_uas = float(uas_num) / nb_tot
    score_las = float(las_num) / nb_tot
    score_ls = float(ls_num) / nb_tot  # NEW

    return (score_uas, score_las, score_ls)


# 2016-09-30 undirected variants
def compute_uas_las_undirected(dtree_true, dtree_pred):
    """Compute dependency metrics for trees in dtree_pred wrt dtree_true.

    The computed metrics are the traditional UAS and LAS, plus LS
    for Labelling Score (counts of correct labels, regardless of head).

    Parameters
    ----------
    dtree_true: list of RstDepTree
        Reference trees

    dtree_pred: list of RstDepTree
        Predicted trees

    Returns
    -------
    (uas, las, ls): (float, float, float)
        The Unlabelled and Labelled Attachment Scores, plus the
        Labelling Score (new).
    """
    nb_ua_ok = 0  # correct unlabelled deps
    nb_la_ok = 0  # correct labelled deps
    nb_tot = 0  # total deps

    for dt_true, dt_pred in itertools.izip(dtree_true, dtree_pred):
        # undirected dependencies are equivalent to the span they cover
        # each span is a tuple with a tuple inside ((fst, snd), lbl)
        spans_true = set((tuple(sorted((gov, dep))), lbl)
                         for dep, (gov, lbl)
                         in enumerate(zip(dt_true.heads[1:], dt_true.labels[1:]),
                                      start=1))
        spans_pred = set((tuple(sorted((gov, dep))), lbl)
                         for dep, (gov, lbl)
                         in enumerate(zip(dt_pred.heads[1:], dt_pred.labels[1:]),
                                      start=1))
        nb_tot += len(spans_pred)
        nb_ua_ok += len(set(x[0] for x in spans_true).intersection(
            set(x[0] for x in spans_pred)))
        nb_la_ok += len(spans_true.intersection(spans_pred))

    score_uas = float(nb_ua_ok) / nb_tot
    score_las = float(nb_la_ok) / nb_tot

    return (score_uas, score_las)
