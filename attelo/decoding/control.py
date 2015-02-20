"""
Central interface to the decoders
"""

from __future__ import print_function
from enum import Enum
import sys

from attelo.learning import (can_predict_proba)
from attelo.report import Count
from attelo.table import (for_attachment, for_labelling, UNRELATED)
from attelo.util import truncate
from .util import (DecoderException)
# pylint: disable=too-few-public-methods


class DecodingMode(Enum):
    '''
    How to do decoding:

        * joint: predict attachment/relations together
        * post_label: predict attachment, then independently
                      predict relations on resulting graph
    '''
    joint = 1
    post_label = 2

# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


def _pick_attached_prob(model):
    '''
    Given a model, return a function that picks out the probability
    of attachment out of a distribution. In the usual case, the
    distribution consists of probabilities for unattached, and
    attached respectively; but we want account for the corner cases
    where either everything or nothing is attached.

    :rtype [float] -> float
    '''
    try:
        idx = list(model.classes_).index(1)
        return lambda dist: dist[idx]
    except ValueError:
        # never atached...
        return lambda _: 0.


def _combine_probs(dpack, models, debug=False):
    """for all EDU pairs, retrieve probability of the best relation
    on that pair, given the probability of an attachment

    """
    pick_attach = _pick_attached_prob(models.attach)
    pick_relate = max

    def link(pair, a_probs, r_probs, label):
        'return a combined-probability link'
        edu1, edu2 = pair
        # TODO: log would be better, no?
        prob = pick_attach(a_probs) * pick_relate(r_probs)
        if debug:
            print('DECODE', edu1.id, edu2.id, file=sys.stderr)
            print(' edu1: ', truncate(edu1.text, 50), file=sys.stderr)
            print(' edu2: ', truncate(edu2.text, 50), file=sys.stderr)
            print(' attach: ', a_probs, pick_attach(a_probs), file=sys.stderr)
            print(' relate: ', r_probs, pick_relate(r_probs), file=sys.stderr)
            print(' combined: ', prob, file=sys.stderr)
        return (edu1, edu2, prob, label)

    attach_pack = for_attachment(dpack)
    relate_pack = for_labelling(dpack)

    attach_probs = models.attach.predict_proba(attach_pack.data)
    relate_probs = models.relate.predict_proba(relate_pack.data)
    relate_idxes = models.relate.predict(relate_pack.data)
    relate_labels = [relate_pack.get_label(i) for i in relate_idxes]

    # pylint: disable=star-args
    return [link(*x) for x in
            zip(dpack.pairings, attach_probs, relate_probs, relate_labels)]
    # pylint: disable=star-args


def _add_labels(dpack, models, predictions):
    """given a list of predictions, predict labels for a given set of edges
    (=post-labelling an unlabelled decoding)

    :param pack: data pack

    :type predictions: [prediction] (see `attelo.decoding.interface`)
    :rtype: [prediction]
    """

    relate_pack = for_labelling(dpack)
    relate_idxes = models.relate.predict(relate_pack.data)
    relate_labels = [relate_pack.get_label(i) for i in relate_idxes]
    label_dict = {(edu1.id, edu2.id): label
                  for (edu1, edu2), label in
                  zip(dpack.pairings, relate_labels)}

    def update(link):
        '''replace the link label (the original by rights is something
        like "unlabelled"'''
        edu1, edu2, _ = link
        label = label_dict[(edu1, edu2)]
        return (edu1, edu2, label)

    res = []
    for pred in predictions:
        res.append(update(p) for p in pred)
    return res


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


def _get_attach_only_prob(dpack, models):
    """
    Attachment probabilities (only) for each EDU pair in the data
    """
    pack = for_attachment(dpack)
    if can_predict_proba(models.attach):
        confidence = models.attach.predict_proba(dpack.data)
        pick_attach = _pick_attached_prob(models.attach)
    else:
        confidence = models.attach.decision_function(dpack.data)
        pick_attach = lambda x: x

    def link(pair, dist):
        ':rtype: proposed link (see attelo.decoding.interface)'
        id1, id2 = pair
        conf = pick_attach(dist)
        return (id1, id2, conf, 'unlabelled')

    return [link(x, y) for x, y in zip(pack.pairings, confidence)]


def decode(mode, decoder, dpack, models):
    """
    Decode every instance in the attachment table (predicting
    relations too if we have the data/model for it).
    Return the predictions made

    :type models: Team(model)
    """

    if mode != DecodingMode.post_label:
        if not can_predict_proba(models.attach):
            oops = ('Attachment model does not know how to predict '
                    'probabilities. It should only be used in post '
                    'labelling mode')
            raise DecoderException(oops)
        if not can_predict_proba(models.relate):
            raise DecoderException('Relation labelling model does not '
                                   'know how to predict probabilities')

        prob_distrib = _combine_probs(dpack, models)
    else:
        prob_distrib = _get_attach_only_prob(dpack, models)
    # print prob_distrib

    predictions = decoder.decode(prob_distrib)

    if mode == DecodingMode.post_label:
        return _add_labels(dpack, models, predictions)
    else:
        return predictions


def count_correct(dpack, predicted):
    """basic eval: counting correct predicted edges (labelled or not)
    data contains the reference attachments
    labels the corresponding relations

    :rtype: :py:class:`attelo.report.Count`
    """
    tpos_attach = 0
    tpos_label = 0
    dict_predicted = {(arg1, arg2): rel for arg1, arg2, rel in predicted
                      if rel != UNRELATED}
    pack = dpack.attached_only()
    for edu_pair, ref_label in zip(pack.pairings, pack.target):
        edu1, edu2 = edu_pair
        pred_label = dict_predicted.get((edu1.id, edu2.id))
        if pred_label is not None:
            tpos_attach += 1
            if dpack.label_number(pred_label) == ref_label:
                tpos_label += 1

    return Count(tpos_attach=tpos_attach,
                 tpos_label=tpos_label,
                 tpos_fpos=len(dict_predicted.keys()),
                 tpos_fneg=len(dpack.attached_only().pairings))
