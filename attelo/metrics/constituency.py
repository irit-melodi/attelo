"""Metrics for constituency trees.

TODO
----
* [ ] factor out the report from the parseval function, see
`sklearn.metrics.classification.classification_report`
* [ ] refactor the selection functions that enable to break down
evaluations, to avoid almost duplicates (as currently)
"""

from __future__ import print_function

import numpy as np

from .classification_structured import (precision_recall_fscore_support,
                                        unique_labels)
from .util import get_spans


# label extraction functions
LBL_FNS = [
    ('S', lambda span: span[0][1] - span[0][0]),  # scores by span length
    ('S+N', lambda span: span[1]),
    ('S+R', lambda span: span[2]),
    ('S+N+R', lambda span: '{}-{}'.format(span[2], span[1])),
]


def discourse_parseval_scores(ctree_true, ctree_pred, lbl_fn,
                              labels=None, average=None):
    """Compute discourse PARSEVAL scores for ctree_pred wrt ctree_true.

    Parameters
    ----------
    ctree_true : list of list of RSTTree or SimpleRstTree

    ctree_pred : list of list of RSTTree or SimpleRstTree

    labels : list of string, optional
        Corresponds to sklearn's target_names IMO

    Returns
    -------
    precision : float (if average is not None) or array of float, shape =\
        [n_unique_labels]
        Weighted average of the precision of each class.

    recall : float (if average is not None) or array of float, shape =\
        [n_unique_labels]

    fbeta_score : float (if average is not None) or array of float, shape =\
        [n_unique_labels]

    support : int (if average is not None) or array of int, shape =\
        [n_unique_labels]
        The number of occurrences of each label in ``ctree_true``.

    References
    ----------
    .. [1] `Daniel Marcu (2000). "The theory and practice of discourse
           parsing and summarization." MIT press.

    """

    # extract descriptions of spans from the true and pred trees
    spans_true = [get_spans(ct_true) for ct_true in ctree_true]
    spans_pred = [get_spans(ct_pred) for ct_pred in ctree_pred]
    # use lbl_fn to define labels
    spans_true = [[(span[0], lbl_fn(span)) for span in spans]
                  for spans in spans_true]
    spans_pred = [[(span[0], lbl_fn(span)) for span in spans]
                  for spans in spans_pred]

    p, r, f, s = precision_recall_fscore_support(spans_true, spans_pred,
                                                 labels=labels,
                                                 average=average)
    return p, r, f, s


def parseval_report(ctree_true, ctree_pred, labels=None, lbl_fn=None,
                    average=None, digits=2):
    """Build a text report showing the PARSEVAL discourse metrics.

    FIXME model after sklearn.metrics.classification.classification_report

    Parameters
    ----------
    ctree_true : list of RSTTree or SimpleRstTree
        Ground truth (correct) target structures.

    ctree_pred : list of RSTTree or SimpleRstTree
        Estimated target structures as predicted by a parser.

    labels : list of string, optional
        Relation labels to include in the evaluation.
        FIXME Corresponds more to target_names in sklearn IMHO.

    lbl_fn : function from tuple((int, int), string, string) to string
        Label extraction function

    digits : int
        Number of digits for formatting output floating point values.

    Returns
    -------
    report : string
        Text summary of the precision, recall, F1 score, support for each
        class (or micro-averaged over all classes).

    References
    ----------
    .. [1] `Daniel Marcu (2000). "The theory and practice of discourse
           parsing and summarization." MIT press.

    """

    if labels is None:
        labels = unique_labels(ctree_true, ctree_pred)

    last_line_heading = 'avg / total'

    width = max(len(lbl) for lbl in labels)
    width = max(width, len(last_line_heading), digits)

    headers = ["precision", "recall", "f1-score", "support"]
    fmt = '%% %ds' % width  # first col: class name
    fmt += '  '
    fmt += ' '.join(['% 9s' for _ in headers])
    fmt += '\n'

    headers = [""] + headers
    report = fmt % tuple(headers)
    report += '\n'

    # call with average=None to compute per-class scores, then
    # compute average here and print it
    p, r, f1, s = discourse_parseval_scores(ctree_true, ctree_pred,
                                            labels=labels,
                                            lbl_fn=lbl_fn,
                                            average=None)
    # one line per label
    for i, label in enumerate(labels):
        values = [label]
        for v in (p[i], r[i], f1[i]):
            values += ["{0:0.{1}f}".format(v, digits)]
        values += ["{0}".format(s[i])]
        if average is None:  # print per-class scores for average=None only
            report += fmt % tuple(values)

    # print only if per-class scores
    if average is None:
        report += '\n'

    # compute averages for the bottom line
    values = [last_line_heading]
    for v in (np.average(p, weights=s),
              np.average(r, weights=s),
              np.average(f1, weights=s)):
        values += ["{0:0.{1}f}".format(v, digits)]
    values += ['{0}'.format(np.sum(s))]
    report += fmt % tuple(values)

    return report


def parseval_reports(ctree_true, ctree_pred, labels=None, average=None,
                     digits=2):
    """Build a text report showing the PARSEVAL discourse metrics.

    FIXME model after sklearn.metrics.classification.classification_report

    Parameters
    ----------
    ctree_true : list of RSTTree or SimpleRstTree
        Ground truth (correct) target structures.

    ctree_pred : list of RSTTree or SimpleRstTree
        Estimated target structures as predicted by a parser.

    labels : list of string, optional
        Relation labels to include in the evaluation.
        FIXME Corresponds more to target_names in sklearn IMHO.

    digits : int
        Number of digits for formatting output floating point values.

    Returns
    -------
    report : string
        Text summary of the precision, recall, F1 score, support for each
        class (or micro-averaged over all classes).

    References
    ----------
    .. [1] `Daniel Marcu (2000). "The theory and practice of discourse
           parsing and summarization." MIT press.

    """
    # extract one report per type of metric
    reports = [(metric_type, parseval_report(ctree_true, ctree_pred,
                                             labels=labels,
                                             lbl_fn=lbl_fn,
                                             average=average,
                                             digits=digits))
               for metric_type, lbl_fn in LBL_FNS]
    return reports
