"""
Central interface to the decoders
"""

from __future__ import print_function
from enum import Enum
import sys

from attelo.learning import (can_predict_proba)
from attelo.report import Count
from attelo.table import (for_attachment, for_labelling,
                          UNRELATED, UNLABELLED)
from attelo.util import truncate
from .intra import (IntraInterPair, select_subgrouping)
from .util import (DecoderException,
                   get_sorted_edus,
                   subgroupings)
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


def _add_labels(dpack, models, predictions, clobber=True):
    """given a list of predictions, predict labels for a given set of edges
    (=post-labelling an unlabelled decoding)

    :param pack: data pack

    :type predictions: [prediction] (see `attelo.decoding.interface`)
    :rtype: [prediction]

    :param clobber: if True, override pre-existing labels; if False, only
                    do so if == UNLABELLED
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
        edu1, edu2, old_label = link
        can_replace = clobber or old_label == UNLABELLED
        label = label_dict[(edu1, edu2)] if can_replace else old_label
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
        return (id1, id2, conf, UNLABELLED)

    return [link(x, y) for x, y in zip(pack.pairings, confidence)]


def build_prob_distrib(mode, dpack, models):
    """
    Extract a decoder probability distribution from the models
    for all instances in the datapack

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

        return _combine_probs(dpack, models)
    else:
        return _get_attach_only_prob(dpack, models)


def _maybe_post_label(mode, dpack, models, predictions,
                      clobber=True):
    """
    If post labelling mode is enabled, apply the best label from
    our relation model to all links in the prediction
    """
    if mode == DecodingMode.post_label:
        return _add_labels(dpack, models, predictions, clobber=clobber)
    else:
        return predictions


def decode(mode, decoder, dpack, models):
    """
    Decode every instance in the attachment table (predicting
    relations too if we have the data/model for it).

    Use intra/inter-sentential decoding if the decoder is a
    :py:class:`IntraInterDecoder` (duck typed). Note that
    you must also supply intra/inter sentential models
    for this

    Return the predictions made.

    :type: models: Team(model) or IntraInterPair(Team(model))
    """
    if callable(getattr(decoder, "decode_sentence", None)):
        func = decode_intra_inter
    else:
        func = decode_vanilla
    return func(mode, decoder, dpack, models)


def decode_vanilla(mode, decoder, dpack, models):
    """
    Decode every instance in the attachment table (predicting
    relations too if we have the data/model for it).
    Return the predictions made

    :type models: Team(model)
    """
    prob_distrib = build_prob_distrib(mode, dpack, models)
    predictions = decoder.decode(prob_distrib)
    return _maybe_post_label(mode, dpack, models, predictions)


def decode_intra_inter(mode, decoder, dpack, models):
    """
    Variant of `decode` which uses an IntraInterDecoder rather than
    a normal decoder

    :type models: IntraInterPair(Team(model))
    """
    prob_distribs =\
        IntraInterPair(intra=build_prob_distrib(mode, dpack, models.intra),
                       inter=build_prob_distrib(mode, dpack, models.inter))
    sorted_edus = get_sorted_edus(prob_distribs.inter)

    # launch a decoder per sentence
    sent_parses = []
    for subg in subgroupings(sorted_edus):
        mini_distrib = select_subgrouping(prob_distribs.intra, subg)
        sent_predictions = decoder.decode_sentence(mini_distrib)
        sent_parses.append(_maybe_post_label(mode, dpack, models.intra,
                                             sent_predictions))
    ##########

    doc_predictions = decoder.decode_document(prob_distribs.inter, sent_parses)
    return _maybe_post_label(mode, dpack, models.inter, doc_predictions,
                             clobber=False)


def count_correct(dpack, predicted):
    """basic eval: counting correct predicted edges (labelled or not)
    data contains the reference attachments
    labels the corresponding relations

    :rtype: :py:class:`attelo.report.Count`
    """
    score_attach = 0
    score_label = 0
    dict_predicted = {(arg1, arg2): rel for arg1, arg2, rel in predicted
                      if rel != UNRELATED}
    pack = dpack.attached_only()
    for edu_pair, ref_label in zip(pack.pairings, pack.target):
        edu1, edu2 = edu_pair
        pred_label = dict_predicted.get((edu1.id, edu2.id))
        if pred_label is not None:
            score_attach += 1
            if dpack.label_number(pred_label) == ref_label:
                score_label += 1

    return Count(correct_attach=score_attach,
                 correct_label=score_label,
                 total_predicted=len(dict_predicted.keys()),
                 total_reference=len(pack.pairings))
