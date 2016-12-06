"""Common metrics on dependency trees.
"""

from __future__ import absolute_import, print_function

import itertools

import numpy as np


def compute_uas_las(dtree_true, dtree_pred, include_ls=True,
                    include_las_n_o_no=False):
    """Compute dependency metrics for trees in dtree_pred wrt dtree_true.

    The computed metrics are the traditional UAS and LAS, plus LS
    for Labelling Score (counts of correct labels, regardless of head).

    Parameters
    ----------
    dtree_true: list of RstDepTree
        Reference trees

    dtree_pred: list of RstDepTree
        Predicted trees

    include_ls: boolean, defauts to True
        If True, the LS metric is computed. This evaluates the accuracy
        of the label of the incoming relation of each EDU.

    include_las_n_o_no: boolean, defaults to False
        If True, the LAS+N, LAS+O, LAS+N+O metrics are computed. These
        include respectively nuclearity, order, and both to the usual
        LAS.

    Returns
    -------
    res: tuple of float
        The Unlabelled and Labelled Attachment Scores, plus the
        Labelling Score (new) if include_ls, plus LAS+N, LAS+O, LAS+N+O
        if include_las_n_o_no.
    """
    nb_ua_ok = 0  # correct unlabelled deps
    nb_la_ok = 0  # correct labelled deps
    if include_ls:
        # correct labellings (right labels, possibly wrong heads)
        nb_l_ok = 0
    if include_las_n_o_no:
        # nuclearity and order
        nb_lan_ok = 0
        nb_lao_ok = 0
        nb_lano_ok = 0
    nb_tot = 0  # total deps

    for dt_true, dt_pred in itertools.izip(dtree_true, dtree_pred):
        # heads and labels are stored as two lists
        # exclude fake root from metrics
        # _true
        heads_true = dt_true.heads[1:]
        labels_true = dt_true.labels[1:]
        # _pred
        heads_pred = dt_pred.heads[1:]
        labels_pred = dt_pred.labels[1:]
        if include_las_n_o_no:
            # LAS + nuclearity and order
            # _true
            nucs_true = dt_true.nucs[1:]
            rnks_true = dt_true.ranks[1:]
            # _pred
            nucs_pred = dt_pred.nucs[1:]
            rnks_pred = dt_pred.ranks[1:]

        for i in range(len(heads_pred)):
            # attachments
            if heads_pred[i] == heads_true[i]:
                nb_ua_ok += 1
                if labels_pred[i] == labels_true[i]:
                    nb_la_ok += 1
                    #
                    if include_las_n_o_no:
                        # LAS + nuclearity and order
                        if nucs_pred[i] == nucs_true[i]:
                            nb_lan_ok += 1
                        if rnks_pred[i] == rnks_true[i]:
                            nb_lao_ok += 1
                        if (nucs_pred[i] == nucs_true[i] and
                            rnks_pred[i] == rnks_true[i]):
                            # both order and nuclearity
                            nb_lano_ok += 1

            # NEW evaluate labelling only
            if include_ls:
                if labels_pred[i] == labels_true[i]:
                    nb_l_ok += 1
                
            # update total
            nb_tot += 1

    score_uas = float(nb_ua_ok) / nb_tot
    score_las = float(nb_la_ok) / nb_tot
    res = [score_uas, score_las]

    if include_ls:
        score_ls = float(nb_l_ok) / nb_tot  # NEW
        res += [score_ls]
    if include_las_n_o_no:
        score_las_n = float(nb_lan_ok) / nb_tot
        score_las_o = float(nb_lao_ok) / nb_tot
        score_las_no = float(nb_lano_ok) / nb_tot
        res += [score_las_n, score_las_o, score_las_no]

    res = tuple(res)
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
