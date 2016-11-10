'''
Scoring decoding results
'''

from __future__ import print_function

from collections import (defaultdict, namedtuple)
import itertools

import numpy as np
from sklearn.metrics import confusion_matrix

# WIP
from educe.rst_dt.annotation import _binarize
from educe.rst_dt.corpus import RstRelationConverter, RELMAP_112_18_FILE
from educe.rst_dt.metrics.rst_parseval import LBL_FNS, rst_parseval_report
# end WIP

from .table import UNRELATED, attached_only, get_label_string
from .metrics.util import get_oracle_ctrees, oracle_ctree_spans


# pylint: disable=too-few-public-methods


class Count(namedtuple('Count',
                       ['tpos_attach',
                        'tpos_label',
                        'tpos_fpos',
                        'tpos_fneg'])):
    """
    Things we would count on dep edges during the scoring process
    """
    @classmethod
    def sum(cls, counts):
        """
        Count made of the total of all counts
        """
        counts = list(counts)  # accept iterable
        return cls(sum(x.tpos_attach for x in counts),
                   sum(x.tpos_label for x in counts),
                   sum(x.tpos_fpos for x in counts),
                   sum(x.tpos_fneg for x in counts))


# WIP
class CSpanCount(namedtuple('CSpanCount',
                            ['tpos',
                             'tpos_fpos',
                             'tpos_fneg'])):
    """
    Things we would count on cspans during the scoring process
    """
    @classmethod
    def sum(cls, counts):
        """
        Count made of the total of all counts
        """
        counts = list(counts)  # accept iterable
        return cls(sum(x.tpos for x in counts),
                   sum(x.tpos_fpos for x in counts),
                   sum(x.tpos_fneg for x in counts))
# end WIP


class EduCount(namedtuple('EduCount',
                          ['correct_attach',
                           'correct_label',
                           'total'])):
    """
    Things we would count during the scoring process
    """
    @classmethod
    def sum(cls, counts):
        """
        Count made of the total of all counts
        """
        return cls(sum(x.correct_attach for x in counts),
                   sum(x.correct_label for x in counts),
                   sum(x.total for x in counts))

    def __add__(self, other):
        return EduCount(self.correct_attach + other.correct_attach,
                        self.correct_label + other.correct_label,
                        self.total + other.total)


def select_in_pack(dpack, predictions):
    """Given some predictions, return only the ones that are
    in the given datapack.

    This could be useful for situations where you want to
    evaluate on parts of a (larger) datapack. If you have
    predictions on the whole pack and you only want
    those which are relevant to the smaller slice, just
    pass in that smaller datapack to get the corresponding
    subset of predictions back
    """
    pairing_ids = {(e1.id, e2.id) for e1, e2 in dpack.pairings}
    return [(id1, id2, r) for id1, id2, r in predictions
            if (id1, id2) in pairing_ids]


def score_edges(dpack, predictions):
    """Count correctly predicted directed and undirected edges and labels.

    Note that undirected label counts are undefined and
    hardcoded to 0

    Parameters
    ----------
    dpack : DataPack
        Datapack containing ground truth edges.

    predictions: list of (string, string, string)
        Predicted edges: (edu1_id, edu2_id, label)

    Returns
    -------
    undirected : attelo.report.Count
        Count for undirected edges.

    directed : attelo.report.Count
        Count for directed edges.
    """
    att_pack, _ = attached_only(dpack, dpack.target)
    dict_predicted = {(arg1, arg2): rel for arg1, arg2, rel in predictions
                      if rel != UNRELATED}

    # undirected
    u_predicted = set(tuple(sorted((arg1, arg2)))
                      for arg1, arg2 in dict_predicted)
    u_gold = set(tuple(sorted((edu1.id, edu2.id)))
                 for edu1, edu2 in att_pack.pairings)
    undirected = Count(tpos_attach=len(u_gold & u_predicted),
                       tpos_label=0,
                       tpos_fpos=len(u_predicted),
                       tpos_fneg=len(u_gold))

    # directed
    tpos_attach = 0
    tpos_label = 0
    for (edu1, edu2), ref_label in zip(att_pack.pairings, att_pack.target):
        pred_label = dict_predicted.get((edu1.id, edu2.id))
        if pred_label is not None:
            tpos_attach += 1
            if att_pack.label_number(pred_label) == ref_label:
                tpos_label += 1
    directed = Count(tpos_attach=tpos_attach,
                     tpos_label=tpos_label,
                     tpos_fpos=len(dict_predicted.keys()),
                     tpos_fneg=len(att_pack.pairings))

    return undirected, directed


def score_cspans(dpacks, dpredictions, coarse_rels=True, binary_trees=True,
                 oracle_ctree_gold=False):
    """Count correctly predicted spans.

    Parameters
    ----------
    dpacks : list of DataPack
        A DataPack per document

    dpredictions : list of ?
        Prediction for each document

    coarse_rels : boolean, optional
        If True, convert relation labels to their coarse-grained version.

    binary_trees : boolean, optional
        If True, convert (gold) constituency trees to their binary version.

    oracle_ctree_gold : boolean, optional
        If True, use oracle gold constituency trees, rebuilt from the
        gold dependency tree. This should emulate the evaluation in (Li 2014).

    Returns
    -------
    cnt_s : Count
        Count S

    cnt_sn : Count
        Count S+N

    cnt_sr : Count
        Count S+R

    cnt_snr : Count
        Count S+N+R
    """
    # trim down DataPacks
    att_packs = [attached_only(dpack, dpack.target)[0]
                 for dpack in dpacks]

    # ctree_gold: oracle (from dependency version) vs true gold
    if oracle_ctree_gold:
        edges_golds = [[(edu1.id, edu2.id, att_pack.get_label(rel))
                        for (edu1, edu2), rel
                        in zip(att_pack.pairings, att_pack.target)
                        if att_pack.get_label(rel) != UNRELATED]
                       for att_pack in att_packs]
        ctree_golds = [get_oracle_ctrees(edges_gold, att_pack.edus)
                       for edges_gold, att_pack
                       in zip(edges_golds, att_packs)]
    else:
        ctree_golds = [dpack.ctarget.values() for dpack in dpacks]
    # WIP coarse-grained rels and binary
    # these probably don't belong here because they leak educe stuff in
    rel_conv = RstRelationConverter(RELMAP_112_18_FILE).convert_tree
    binarize_tree = _binarize
    if coarse_rels:
        ctree_golds = [[rel_conv(ctg) for ctg in ctree_gold]
                       for ctree_gold in ctree_golds]
    if binary_trees:
        ctree_golds = [[binarize_tree(ctg) for ctg in ctree_gold]
                       for ctree_gold in ctree_golds]
    # end WIP
    # spans of the gold constituency trees
    ctree_spans_golds = [list(itertools.chain.from_iterable(
        ctg.get_spans() for ctg in ctree_gold))
                         for ctree_gold in ctree_golds]
    # spans of the predicted oracle constituency trees
    edges_preds = [[(edu1, edu2, rel)
                    for edu1, edu2, rel in predictions
                    if rel != UNRELATED]
                   for predictions in dpredictions]
    ctree_spans_preds = [oracle_ctree_spans(edges_pred, att_pack.edus)
                         for edges_pred, att_pack
                         in zip(edges_preds, att_packs)]

    ctree_true = ctree_golds  # yerk
    ctree_pred = [get_oracle_ctrees(edges_pred, att_pack.edus)
                  for edges_pred, att_pack
                  in zip(edges_preds, att_packs)]
    # 2016-10-02 force one ctree per doc ; we need to reconsider when we
    # do doc-level eval
    for ct_true in ctree_true:
        if len(ct_true) > 1:
            raise NotImplementedError(
                "Currently unable to handle multiple ctrees per doc")
    ctree_true = [ct_true[0] for ct_true in ctree_true]
    for ct_pred in ctree_pred:
        if len(ct_pred) > 1:
            raise NotImplementedError(
                "Currently unable to handle multiple ctrees per doc")
    ctree_pred = [ct_pred[0] for ct_pred in ctree_pred]
    # end force one ctree per doc

    print(rst_parseval_report(ctree_true, ctree_pred))

    # FIXME replace loop with attelo.metrics.constituency.XXX
    cnts = []
    for metric_type, lbl_fn in LBL_FNS:
        y_true = [[(span[0], lbl_fn(span)) for span in ctree_spans]
                  for ctree_spans in ctree_spans_golds]
        y_pred = [[(span[0], lbl_fn(span)) for span in ctree_spans]
                  for ctree_spans in ctree_spans_preds]

        y_tpos = sum(len(set(yt) & set(yp))
                     for yt, yp in zip(y_true, y_pred))
        y_tpos_fpos = sum(len(yp) for yp in y_pred)
        y_tpos_fneg = sum(len(yt) for yt in y_true)
        cnts.append(CSpanCount(tpos=y_tpos,
                               tpos_fpos=y_tpos_fpos,
                               tpos_fneg=y_tpos_fneg))

    return cnts[0], cnts[1], cnts[2], cnts[3]


def score_edus(dpack, predictions):
    """compute the number of edus

    1. with correct attachments to their heads (ie. given edu
    e, every reference (p, e) link is present, and only such
    links are present)
    2. with correct attachments to their heads and labels
    (ie. given edu e, every reference (p, e) link is present,
    with the correct label, and only such links are present)

    This score may quite low if we are predicted a multiheaded
    graph

    :rtype: :py:class:`EduCount`
    """

    e_predictions = defaultdict(list)
    for parent, edu, rel in predictions:
        if rel == UNRELATED:
            continue
        e_predictions[edu].append((parent, dpack.label_number(rel)))

    e_reference = defaultdict(list)
    unrelated = dpack.label_number(UNRELATED)
    for (parent, edu), ref_label in zip(dpack.pairings, dpack.target):
        if ref_label == unrelated:
            continue
        e_reference[edu.id].append((parent.id, int(ref_label)))

    correct_attach = 0
    correct_label = 0
    for edu in dpack.edus:
        pred = sorted(e_predictions.get(edu.id, []))
        ref = sorted(e_reference.get(edu.id, []))
        if [x[0] for x in pred] == [x[0] for x in ref]:
            correct_attach += 1
        if pred == ref:
            correct_label += 1
    assert correct_label <= correct_attach

    return EduCount(correct_attach=correct_attach,
                    correct_label=correct_label,
                    total=len(dpack.edus))


def score_edges_by_label(dpack, predictions):
    """
    Return (as a generator) a list of pairs associating each
    label with scores for that label.

    If you are scoring mutiple folds you could loop over the
    folds, combining pre-existing scores for each label within
    the fold with its counterpart in the other folds
    """
    predictions = [(e1, e2, r) for (e1, e2, r) in predictions
                   if r != UNRELATED]

    for label in dpack.labels:
        if label == UNRELATED:
            continue
        label_num = dpack.label_number(label)
        # pylint: disable=no-member
        r_indices = np.where(dpack.target == label_num)[0]
        # pylint: disable=no-member
        r_dpack = dpack.selected(r_indices)
        r_predictions = [(e1, e2, r) for (e1, e2, r) in predictions
                         if r == label]
        yield label, score_edges(r_dpack, r_predictions)


def build_confusion_matrix(dpack, predictions):
    """return a confusion matrix show predictions vs desired labels
    """
    # first, we need to align target_true and target_pred
    # FIXME avoid this costly operation: make sure that dpack.pairings
    # and dpack.target keep the same ordering as in the .pairings file ;
    # we need to find where this ordering is lost
    lbl_dict_true = {(src.id, tgt.id): lbl for (src, tgt), lbl
                     in zip(dpack.pairings, dpack.target)}
    target_true = [lbl_dict_true[(src, tgt)] for src, tgt, _ in predictions]
    target_pred = [dpack.label_number(label) for _, _, label in predictions]
    # we want the confusion matrices to have the same shape regardless
    # of what labels happen to be used in the particular fold
    # pylint: disable=no-member
    labels = np.arange(0, len(dpack.labels))
    # pylint: enable=no-member
    return confusion_matrix(target_true, target_pred, labels)


def empty_confusion_matrix(dpack):
    """return a zero array that could be used to accumulate future
    confusion matrix results
    """
    llen = len(dpack.labels)
    # pylint: disable=no-member
    return np.zeros((llen, llen), dtype=np.int32)
    # pylint: disable=no-member


def discriminating_features(models, labels, vocab, top_n):
    """return the most discriminating features (and their weights)
    for each label in the models; or None if the model does not
    support this sort of query

    See :py:func:`attelo.report.show_discriminating_features`

    :param top_n number of features to return
    :type top_n: int

    :type models: Team(model)

    :type labels: [string]

    :param sequence of string labels, ie. one for each possible feature
    :type vocab: [string]

    :rtype: [(string, [(string, float)])] or None
    """
    def fmt(features):
        'label important features'
        return [(vocab[f], w) for f, w in features]

    rows = None
    if hasattr(models.attach, 'important_features'):
        imp = models.attach.important_features(top_n)
        if imp is not None:
            rows = rows or []
            rows.append(('(attach)', fmt(imp)))

    if hasattr(models.label, 'important_features_multi'):
        per_label = models.label.important_features_multi(top_n)
        if per_label is not None:
            rows = rows or []
            for lnum in sorted(per_label):
                label = get_label_string(labels, lnum)
                rows.append((label, fmt(per_label[lnum])))
    if hasattr(models.label, 'important_features'):
        imp = models.label.important_features(top_n)
        if imp is not None:
            rows = rows or []
            rows.append(('(label)', fmt(imp)))

    return rows
