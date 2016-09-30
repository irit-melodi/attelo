"""Common metrics on dependency trees.
"""

import itertools

import numpy as np


def compute_uas_las(dtree_true, dtree_pred):
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
    nb_l_ok = 0  # correct labellings (right labels, possibly wrong heads)
    nb_tot = 0  # total deps

    for dt_true, dt_pred in itertools.izip(dtree_true, dtree_pred):
        # heads and labels are stored as two lists
        # exclude fake root from metrics
        heads_true = dt_true.heads[1:]
        labels_true = dt_true.labels[1:]

        heads_pred = dt_pred.heads[1:]
        labels_pred = dt_pred.labels[1:]

        for i in range(len(heads_pred)):
            # attachments
            if heads_pred[i] == heads_true[i]:
                nb_ua_ok += 1
                if labels_pred[i] == labels_true[i]:
                    nb_la_ok += 1
            # NEW evaluate labelling only
            if labels_pred[i] == labels_true[i]:
                nb_l_ok += 1
            nb_tot += 1

    score_uas = float(nb_ua_ok) / nb_tot
    score_las = float(nb_la_ok) / nb_tot
    score_ls = float(nb_l_ok) / nb_tot  # NEW

    return (score_uas, score_las, score_ls)


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
