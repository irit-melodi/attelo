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
    ('S', lambda span: 1),
    ('S+N', lambda span: span[1]),
    ('S+R', lambda span: span[2]),
    ('S+N+R', lambda span: '{}-{}'.format(span[2], span[1])),
]


# PARSEVAL metrics adapted to the evaluation of discourse parsers,
# with options to get meaningful variants in specific settings
def discourse_parseval_scores(ctree_true, ctree_pred,
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


def parseval_report(ctree_true, ctree_pred, metric_types=None, digits=4,
                    print_support_pred=True, span_sel=None,
                    stringent=False):
    """Build a text report showing the PARSEVAL discourse metrics.

    This is the simplest report we need to generate, it corresponds
    to the arrays of results from the literature.
    Metrics are calculated globally (average='micro').

    Parameters
    ----------
    metric_types: list of strings, optional
        Metrics that need to be included in the report ; if None is
        given, defaults to ['S', 'S+N', 'S+R', 'S+N+R'].
    """
    # select metrics and the corresponding functions
    if metric_types is None:
        metric_types = ['S', 'S+N', 'S+R', 'S+N+R']
    if set(metric_types) - set(x[0] for x in LBL_FNS):
        raise ValueError('Unknown metric types in {}'.format(metric_types))
    metric2lbl_fn = dict(LBL_FNS)

    # prepare report
    width = max(len(str(x)) for x in metric_types)
    width = max(width, digits)

    headers = ["precision", "recall", "f1-score", "support", "sup_pred"]
    fmt = '%% %ds' % width  # first col: class name
    fmt += '  '
    fmt += ' '.join(['% 9s' for _ in headers])
    fmt += '\n'

    headers = [""] + headers
    report = fmt % tuple(headers)
    report += '\n'
    # end prepare report

    # FIXME refactor in tandem with discourse_parseval_scores, to
    # get a coherent and non-redundant API
    # extract descriptions of spans from the true and pred trees
    spans_true = [get_spans(ct_true) for ct_true in ctree_true]
    spans_pred = [get_spans(ct_pred) for ct_pred in ctree_pred]

    metric_scores = dict()
    for metric_type in metric_types:
        lbl_fn = metric2lbl_fn[metric_type]
        # possibly filter data
        sp_true = spans_true
        sp_pred = spans_pred
        if span_sel == 'leaves':
            # only leaf nodes
            sp_true = [[xi for xi in x if xi[0][1] == xi[0][0]]
                       for x in sp_true]
            sp_pred = [[xi for xi in x if xi[0][1] == xi[0][0]]
                       for x in sp_pred]
        elif span_sel == 'non-leaves':
            # only internal nodes
            sp_true = [[xi for xi in x if xi[0][1] != xi[0][0]]
                       for x in sp_true]
            sp_pred = [[xi for xi in x if xi[0][1] != xi[0][0]]
                       for x in sp_pred]
            
        if stringent and False and metric_type in ['S+R', 'S+N+R']:
                # * S+R, S+N+R: exclude 'span'
                sp_true = [[xi for xi in x if xi[2] != 'span']
                           for x in sp_true]
                sp_pred = [[xi for xi in x if xi[2] != 'span']
                           for x in sp_pred]
        # end filter
        y_true = [[(span[0], lbl_fn(span)) for span in spans]
                  for spans in sp_true]
        y_pred = [[(span[0], lbl_fn(span)) for span in spans]
                  for spans in sp_pred]
        # calculate metric
        p, r, f1, s_true, s_pred = precision_recall_fscore_support(
            y_true, y_pred, average='micro')
        metric_scores[metric_type] = (p, r, f1, s_true, s_pred)

    # report
    for metric_type in metric_types:
        (p, r, f1, s_true, s_pred) = metric_scores[metric_type]
        values = [metric_type]
        for v in (p, r, f1):
            values += ["{0:0.{1}f}".format(v, digits)]
        values += ["{0}".format(s_true)]  # support_true
        values += ["{0}".format(s_pred)]  # support_pred
        report += fmt % tuple(values)

    return report


def parseval_detailed_report(ctree_true, ctree_pred,
                             metric_type='S+R',
                             span_sel=None,
                             labels=None,
                             average=None,
                             sort_by_support=True,
                             digits=4):
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
    metric2lbl_fn = dict(LBL_FNS)
    lbl_fn = metric2lbl_fn[metric_type]

    # extract descriptions of spans from the true and pred trees
    spans_true = [get_spans(ct_true) for ct_true in ctree_true]
    spans_pred = [get_spans(ct_pred) for ct_pred in ctree_pred]

    # possibly filter data
    sp_true = spans_true
    sp_pred = spans_pred
    if span_sel == 'leaves':
        # only leaf nodes
        sp_true = [[xi for xi in x if xi[0][1] == xi[0][0]]
                   for x in sp_true]
        sp_pred = [[xi for xi in x if xi[0][1] == xi[0][0]]
                   for x in sp_pred]
    elif span_sel == 'non-leaves':
        # only internal nodes
        sp_true = [[xi for xi in x if xi[0][1] != xi[0][0]]
                   for x in sp_true]
        sp_pred = [[xi for xi in x if xi[0][1] != xi[0][0]]
                   for x in sp_pred]

    # use lbl_fn to extract the label of interest
    y_true = [[(span[0], lbl_fn(span)) for span in spans]
              for spans in sp_true]
    y_pred = [[(span[0], lbl_fn(span)) for span in spans]
              for spans in sp_pred]

    present_labels = unique_labels(y_true, y_pred)

    if labels is None:
        labels = present_labels
        n_labels = None
    else:
        # currently not tested
        n_labels = len(labels)
        labels = np.hstack([labels, np.setdiff1d(present_labels, labels,
                                                 assume_unique=True)])

    last_line_heading = 'avg / total'

    width = max(len(str(lbl)) for lbl in labels)
    width = max(width, len(last_line_heading), digits)

    headers = ["precision", "recall", "f1-score", "support", "sup_pred"]
    fmt = '%% %ds' % width  # first col: class name
    fmt += '  '
    fmt += ' '.join(['% 9s' for _ in headers])
    fmt += '\n'

    headers = [""] + headers
    report = fmt % tuple(headers)
    report += '\n'

    # call with average=None to compute per-class scores, then
    # compute average here and print it
    p, r, f1, s_true, s_pred = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=average)
    sorted_ilbls = enumerate(labels)
    if sort_by_support:
        sorted_ilbls = sorted(sorted_ilbls, key=lambda x: s_true[x[0]],
                              reverse=True)
    # one line per label
    for i, label in sorted_ilbls:
        values = [label]
        for v in (p[i], r[i], f1[i]):
            values += ["{0:0.{1}f}".format(v, digits)]
        values += ["{0}".format(s_true[i])]
        values += ["{0}".format(s_pred[i])]
        if average is None:  # print per-class scores for average=None only
            report += fmt % tuple(values)

    # print only if per-class scores
    if average is None:
        report += '\n'

    # compute averages for the bottom line
    values = [last_line_heading]
    for v in (np.average(p, weights=s_true),
              np.average(r, weights=s_true),
              np.average(f1, weights=s_true)):
        values += ["{0:0.{1}f}".format(v, digits)]
    values += ['{0}'.format(np.sum(s_true))]
    values += ['{0}'.format(np.sum(s_pred))]
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
    reports = []
    for metric_type, lbl_fn in LBL_FNS:
        lbls = labels if metric_type in ['S+R', 'S+N+R'] else None
        reports.append((metric_type,
                        parseval_report(ctree_true, ctree_pred, lbl_fn,
                                        labels=lbls,
                                        average=average,
                                        digits=digits)))
    return reports
