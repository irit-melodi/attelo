"""Metrics for constituency trees.

TODO
----
* [ ] factor out the report from the parseval function, see
`sklearn.metrics.classification.classification_report`
* [ ] refactor the selection functions that enable to break down
evaluations, to avoid almost duplicates (as currently)
"""

from __future__ import print_function

from collections import Counter
import itertools
from itertools import chain

import numpy as np


# util functions
def _unique_span_length(y):
    """Get the set of unique span length in y.

    This assumes that elements of y are pairs of indices that describe
    spans.
    """
    res = set(yi[1] - yi[0] for yi in chain.from_iterable(y))
    return res


def _unique_lbl1(y):
    """Get the set of unique labels in y.

    This assumes that elements of y are tuples where the label is at
    index 1.
    """
    res = set(yi[1] for yi in chain.from_iterable(y))
    return res


def _unique_lbl2(y):
    """Get the set of unique labels in y.

    This assumes that elements of y are tuples where the label is at
    index 2.
    """
    res = set(yi[2] for yi in chain.from_iterable(y))
    return res


_FN_UNIQUE_LABELS = {
    's': _unique_span_length,
    's+n': _unique_lbl1,
    's+r': _unique_lbl1,
    's+n+r': _unique_lbl2,
}


def unique_labels(elt_type, *ys):
    """Extract an ordered array of unique labels.

    Parameters
    ----------
    elt_type: string
        Type of each element, determines how to find the label

    See also
    --------
    `sklearn.utils.multiclass.unique_labels`
    """
    _unique_labels = _FN_UNIQUE_LABELS.get(elt_type, None)
    if not _unique_labels:
        raise ValueError('Unknown element type: ' + elt_type)

    ys_labels = set(chain.from_iterable(_unique_labels(y) for y in ys))
    # TODO check the set of labels contains a unique (e.g. string) type
    # of values
    return np.array(sorted(ys_labels))


def precision_recall_fscore_support(y_true, y_pred, labels=None,
                                    average=None,
                                    elt_type='s'):
    """Compute precision, recall, F-measure and support for each class.

    The support is the number of occurrences of each class in
    ``y_true``.

    Parameters
    ----------
    y_true: list of iterable
        Ground truth target trees, encoded in a sparse format (e.g. list
        of edges or constituent descriptions).

    y_pred: list of iterable
        Estimated targets as returned by a classifier with
        tree-structured outputs. Each tree is encoded in a sparse format
        (e.g. list of either constituents or edges).

    labels: list, optional
        The set of labels to include, and their order if ``average is
        None``.

    average: string, [None (default), 'binary', 'micro', 'macro']
        If ``None``, the scores for each class are returned. Otherwise,
        this determines the type of averaging performed on the data:

        ``'binary'``:
            Only report results for the positive class.
            This is applicable only if targets are binary.
        ``'micro'``:
            Calculate metrics globally by counting the total true
            positives, false negatives and false positives.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean. This does not take label imbalance into account.

    elt_type: string, ['s' (default), 's+n', 's+r', 's+n+r']
        Type of each element of each yi of each y

    Returns
    -------
    precision: float (if average is not None) or array of float, shape=\
        [n_unique_labels]

    recall: float (if average is not None) or array of float, shape=\
        [n_unique_labels]

    fscore: float (if average is not None) or array of float, shape=\
        [n_unique_labels]

    support: int (if average is not None) or array of int, shape=\
        [n_unique_labels]
        The number of occurrences of each label in ``ctree_true``.
    """
    average_options = frozenset([None, 'micro', 'macro'])
    if average not in average_options:
        raise ValueError('average has to be one of' +
                         str(average_options))
    # TEMPORARY
    if average not in set(['micro', None]):
        raise NotImplementedError('average currently has to be micro or None')

    # gather an ordered list of unique labels from y_true and y_pred
    present_labels = unique_labels(elt_type, y_true, y_pred)

    if labels is None:
        labels = present_labels
        n_labels = None
    else:
        # EXPERIMENTAL
        labels = [lbl for lbl in labels if lbl in present_labels]
        n_labels = len(labels)
        # FIXME complete/fix this
        # raise ValueError('Parameter `labels` is currently unsupported')
        # end EXPERIMENTAL

    # compute tp_sum, pred_sum, true_sum
    # true positives for each tree
    tp = [set(yi_true) & set(yi_pred)
          for yi_true, yi_pred in itertools.izip(y_true, y_pred)]

    # TODO refactor to share code with _FN_UNIQUE_LABELS
    if elt_type == 's':
        _lbl_fn = lambda yi: yi[1] - yi[0]
    elif elt_type == 's+n':
        _lbl_fn = lambda yi: yi[1]
    elif elt_type == 's+r':
        _lbl_fn = lambda yi: yi[1]
    elif elt_type == 's+n+r':
        _lbl_fn = lambda yi: yi[2]
    else:
        raise ValueError('elt_type {} not supported'.format(elt_type))

    # TODO find a nicer and faster design that resembles sklearn's, e.g.
    # use np.bincount
    #
    # collect using Counter
    tp_sum = Counter(_lbl_fn(yi) for yi in chain.from_iterable(tp))
    true_sum = Counter(_lbl_fn(yi) for yi in chain.from_iterable(y_true))
    pred_sum = Counter(_lbl_fn(yi) for yi in chain.from_iterable(y_pred))
    # transform to np arrays of floats
    tp_sum = np.array([float(tp_sum[lbl]) for lbl in labels])
    true_sum = np.array([float(true_sum[lbl]) for lbl in labels])
    pred_sum = np.array([float(pred_sum[lbl]) for lbl in labels])

    # TODO rewrite to compute by summing over scores broken down by label
    if average == 'micro':
        tp_sum = np.array([tp_sum.sum()])
        true_sum = np.array([true_sum.sum()])
        pred_sum = np.array([pred_sum.sum()])
        if False:  # extra checks  # false when labels is not None
            ctrl_tp_sum = sum(len(tp_i) for tp_i in tp)
            assert ctrl_tp_sum == tp_sum
            ctrl_true_sum = sum(len(yi_true) for yi_true in y_true)
            assert ctrl_true_sum == true_sum
            ctrl_pred_sum = sum(len(yi_pred) for yi_pred in y_pred)
            assert ctrl_pred_sum == pred_sum

    # finally compute the desired statistics
    # when the div denominator is 0, assign 0.0 (instead of np.inf)
    precision = tp_sum / pred_sum
    precision[pred_sum == 0] = 0.0

    recall = tp_sum / true_sum
    recall[true_sum == 0] = 0.0

    f_score = 2 * (precision * recall) / (precision + recall)
    f_score[precision + recall == 0] = 0.0

    return precision, recall, f_score, true_sum


def compute_parseval_scores(ctree_true, ctree_pred, average=None,
                            exclude_rel_span=False):
    """Compute (and display) PARSEVAL scores for ctree_pred wrt ctree_true.

    Parameters
    ----------
    ctree_true: dict(string, SimpleRstTree)

    ctree_pred: dict(string, SimpleRstTree)

    average: TODO complete using the above function

    Returns
    -------
    FIXME return scores

    References
    ----------
    .. [1] `Daniel Marcu (2000). "The theory and practice of discourse
           parsing and summarization." MIT press.

    """
    labels = None  # FIXME check this is the right default value
    # coarse labels except for span
    # FIXME pass as parameters or move elsewhere ; this might be too
    # RST-dependent
    clabels_wo_span = [
        'attribution',
        'background',
        'cause',
        'comparison',
        'condition',
        'contrast',
        'elaboration',
        'enablement',
        'evaluation',
        'explanation',
        'joint',
        'manner-means',
        'same-unit',
        'summary',
        'temporal',
        'textual',
        'topic-change',
        'topic-comment',
    ]

    # collect all constituents, i.e. all treenodes except for the root
    # node (as is done in Marcu's 2000 book and Joty's eval script)
    tns_true = [[subtree.label()  # was: educe.internalutil.treenode(subtree)
                 for root_child in ct_true
                 for subtree in root_child.subtrees()]
                for ct_true in ctree_true]
    # extract the minimally relevant description of each constituent
    snr_true = [[(tn.edu_span, tn.nuclearity, tn.rel)
                 for tn in tns]
                for tns in tns_true]
    # same for pred
    tns_pred = [[subtree.label()  # was: educe.internalutil.treenode(subtree)
                 for root_child in ct_pred
                 for subtree in root_child.subtrees()]
                for ct_pred in ctree_pred]
    snr_pred = [[(tn.edu_span, tn.nuclearity, tn.rel)
                 for tn in tns]
                for tns in tns_pred]

    # we need 4 different metrics: S, S+N, S+R, S+N+R
    # spans
    s_true = [[c[0] for c in cs]
              for cs in snr_true]
    s_pred = [[c[0] for c in cs]
              for cs in snr_pred]
    # spans + nuclearity
    sn_true = [[(c[0], c[1]) for c in cs]
               for cs in snr_true]
    sn_pred = [[(c[0], c[1]) for c in cs]
               for cs in snr_pred]
    # spans + relation
    sr_true = [[(c[0], c[2]) for c in cs]
               for cs in snr_true]
    sr_pred = [[(c[0], c[2]) for c in cs]
               for cs in snr_pred]
    # spans + nuclearity + relation
    # already computed, see above

    # display
    # TODO don't do this here + automate/loop
    # calculate metrics globally
    target_names = unique_labels('s', s_true, s_pred)
    precision, recall, f_score, support = precision_recall_fscore_support(
        s_true, s_pred, labels=labels, average=average, elt_type='s')
    print('\tP\tR\tF\tsupport')
    print('============================================')
    print('S')
    print('\n'.join('{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.0f}'.format(l, p, r, f, s)
                    for l, p, r, f, s in itertools.izip(
                            target_names,
                            precision, recall, f_score, support)))
    print('--------------------------------------------')
    target_names = unique_labels('s+n', sn_true, sn_pred)
    precision, recall, f_score, support = precision_recall_fscore_support(
        sn_true, sn_pred, labels=labels, average=average, elt_type='s+n')
    print('S+N')
    print('\n'.join('{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.0f}'.format(l, p, r, f, s)
                    for l, p, r, f, s in itertools.izip(
                            target_names,
                            precision, recall, f_score, support)))
    print('--------------------------------------------')
    target_names = unique_labels('s+r', sr_true, sr_pred)
    if exclude_rel_span:
        labels = clabels_wo_span
        target_names = [lbl for lbl in target_names if lbl != 'span']
    precision, recall, f_score, support = precision_recall_fscore_support(
    sr_true, sr_pred, labels=labels, average=average, elt_type='s+r')
    print('S+R')
    print('\n'.join('{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.0f}'.format(l, p, r, f, s)
                    for l, p, r, f, s in itertools.izip(
                            target_names,
                            precision, recall, f_score, support)))
    print('--------------------------------------------')
    target_names = unique_labels('s+n+r', snr_true, snr_pred)
    if exclude_rel_span:
        labels = clabels_wo_span
        target_names = [lbl for lbl in target_names if lbl != 'span']
    precision, recall, f_score, support = precision_recall_fscore_support(
        snr_true, snr_pred, labels=labels, average=average, elt_type='s+n+r')
    print('S+N+R')
    print('\n'.join('{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.0f}'.format(l, p, r, f, s)
                    for l, p, r, f, s in itertools.izip(
                            target_names,
                            precision, recall, f_score, support)))
    print('--------------------------------------------')
