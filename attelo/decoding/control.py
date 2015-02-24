"""
Central interface to the decoders
"""

from __future__ import print_function
from collections import defaultdict
from enum import Enum
import sys

import numpy as np

from attelo.learning import (can_predict_proba)
from attelo.report import (Count, EduCount)
from attelo.table import (for_attachment, for_labelling, for_intra,
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


def _predict_attach(dpack, models):
    """
    Return an array either of probabilities (or in the case of
    non-probability-capable models), confidence scores
    """
    if models.attach == 'oracle':
        to_prob = lambda x: 1.0 if x == 1.0 else 0.0
        # pylint: disable=no-member
        return np.vectorize(to_prob)(dpack.target)
        # pylint: enable=no-member
    elif can_predict_proba(models.attach):
        attach_idx = list(models.attach.classes_).index(1)
        probs = models.attach.predict_proba(dpack.data)
        res = probs[:, attach_idx]
        return res
    else:
        return models.attach.decision_function(dpack.data)


def _predict_relate(dpack, models):
    """
    Return an array of probabilities (that of the best label),
    and an list of labels
    """
    if models.relate == 'oracle':
        idxes = dpack.target
        # pylint: disable=no-member
        probs = np.ones(idxes.shape)
        # pylint: enable=no-member
    elif not can_predict_proba(models.relate):
        raise DecoderException('Tried to use a non-prob decoder for relations')
    else:
        all_probs = models.relate.predict_proba(dpack.data)
        # get the probability associated with the best label
        # pylint: disable=no-member
        probs = np.amax(all_probs, axis=1)
        # pylint: enable=no-member
        idxes = models.relate.predict(dpack.data)
    # pylint: disable=no-member
    get_label = np.vectorize(dpack.get_label)
    # pylint: enable=no-member
    return probs, get_label(idxes)


def _combine_probs(dpack, models, debug=False):
    """for all EDU pairs, retrieve probability of the best relation
    on that pair, given the probability of an attachment

    """
    def link(pair, a_prob, r_prob, label):
        'return a combined-probability link'
        edu1, edu2 = pair
        # TODO: log would be better, no?
        prob = a_prob * r_prob
        if debug:
            print('DECODE', edu1.id, edu2.id, file=sys.stderr)
            print(' edu1: ', truncate(edu1.text, 50), file=sys.stderr)
            print(' edu2: ', truncate(edu2.text, 50), file=sys.stderr)
            print(' attach: ', a_prob, file=sys.stderr)
            print(' relate: ', r_prob, file=sys.stderr)
            print(' combined: ', prob, file=sys.stderr)
        return (edu1, edu2, prob, label)

    attach_pack = for_attachment(dpack)
    relate_pack = for_labelling(dpack)
    attach_probs = _predict_attach(attach_pack, models)
    relate_probs, relate_labels = _predict_relate(relate_pack, models)

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
    _, relate_labels = _predict_relate(relate_pack, models)
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
        attach_pack = for_attachment(dpack)
        pairings = attach_pack.pairings
        # FIXME: this is a bug in how we're calling perceptrons
        # should be just pack
        confidence = _predict_attach(attach_pack, models)
        return [(id1, id2, conf, UNLABELLED) for
                (id1, id2), conf in zip(pairings, confidence)]


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
    if models.intra == 'oracle':
        # intrasentential oracle should be fed with gold sentence roots
        dpacks = IntraInterPair(intra=for_intra(dpack),
                                inter=dpack)
    else:
        # otherwise we really need to bother
        dpacks = IntraInterPair(intra=dpack, inter=dpack)

    prob_distribs =\
        IntraInterPair(intra=build_prob_distrib(mode,
                                                dpacks.intra,
                                                models.intra),
                       inter=build_prob_distrib(mode,
                                                dpacks.inter,
                                                models.inter))
    sorted_edus = get_sorted_edus(prob_distribs.inter)

    # launch a decoder per sentence
    sent_parses = []
    for subg in subgroupings(sorted_edus):
        mini_distrib = select_subgrouping(prob_distribs.intra, subg)
        sent_predictions = decoder.decode_sentence(mini_distrib)
        sent_parses.append(_maybe_post_label(mode, dpacks.intra, models.intra,
                                             sent_predictions))
    ##########

    doc_predictions = decoder.decode_document(prob_distribs.inter, sent_parses)
    return _maybe_post_label(mode, dpacks.inter, models.inter, doc_predictions,
                             clobber=False)


def count_correct_edges(dpack, predicted):
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


def count_correct_edus(dpack, predicted):
    """compute the number of edus

    1. with correct attachments to their heads (ie. given edu
    e, every reference (p, e) link is present, and only such
    links are present)
    2. with correct attachments to their heads and labels
    (ie. given edu e, every reference (p, e) link is present,
    with the correct label, and only such links are present)

    This score may quite low if we are predicted a multiheaded
    graph

    :rtype EduCount
    """

    e_predictions = defaultdict(list)
    for parent, edu, rel in predicted:
        if rel == UNRELATED:
            continue
        e_predictions[edu].append((parent, dpack.label_number(rel)))

    e_reference = defaultdict(list)
    unrelated = dpack.label_number(UNRELATED)
    for edu_pair, ref_label in zip(dpack.pairings, dpack.target):
        if ref_label == unrelated:
            continue
        parent, edu = edu_pair
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
