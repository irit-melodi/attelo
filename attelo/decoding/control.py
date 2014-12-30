"""
Central interface to the decoders
"""

from __future__ import print_function
from collections import namedtuple
from Orange.classification import Classifier
import sys

from attelo.edu import mk_edu_pairs
from attelo.learning.perceptron import is_perceptron_model
from attelo.table import index_by_metas, select_edu_pair
from attelo.report import Count
from .util import DecoderException


# pylint: disable=too-few-public-methods
class DecoderConfig(namedtuple("DecoderConfig",
                               ["phrasebook",
                                "threshold",
                                "post_labelling",
                                "use_prob"])):
    """
    Parameters needed by decoder.
    """
    def __new__(cls,
                phrasebook,
                threshold=None,
                post_labelling=False,
                use_prob=True):
        sup = super(DecoderConfig, cls)
        return sup.__new__(cls,
                           phrasebook=phrasebook,
                           threshold=threshold,
                           post_labelling=post_labelling,
                           use_prob=use_prob)


class DataAndModel(namedtuple("_DataAndModel", "data model")):
    """
    Tuple of a data table accompanied with a model for the kind of
    data within
    """
    pass
# pylint: enable=too-few-public-methods

# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


def _get_inst_attach_orange(model, inst):
    """
    Return the probability of attachment for a single instance,
    given an attachment model.

    :rtype: float
    """
    dist = model(inst, Classifier.GetProbabilities)
    return dist['True'] if 'True' in dist else 0.


def _get_inst_relate_orange(model, inst):
    """
    Return the best label for an instance and the associated
    probability (ie. probability of that label being chosen
    among others)

    :rtype: (float, string)
    """
    label, probs = model(inst, Classifier.GetBoth)
    return max(probs), label.value


def _combine_single_prob(attach, relate, att, rel):
    """return the best relation for a given EDU pair and its
    joint probability with the pair being attached

    helper for _combine_prob

    :rtype (float, string)
    """
    if is_perceptron_model(attach.model):
        if not attach.model.use_prob == True:
            raise DecoderException("ERROR: Trying to output probabilities "
                                   "while Perceptron parametrized with "
                                   "use_prob=False!")
        p_attach = attach.model.get_scores([att])[0][2]
    else:
        p_attach = _get_inst_attach_orange(attach.model, att)

    rel_prob, best_rel = _get_inst_relate_orange(relate.model, rel)

    return (p_attach * rel_prob, best_rel)


def _combine_probs(phrasebook, attach, relate):
    """for all EDU pairs, retrieve probability of the best relation
    on that pair, given the probability of an attachment

    :type attach: DataAndModel
    :type relate: DataAndModel
    """
    # !! instances set must correspond to same edu pair in the same order !!
    distrib = []

    edu_pair = mk_edu_pairs(phrasebook, attach.data.domain)
    attach_instances = sorted(attach.data, key=lambda x: x.get_metas())
    relate_instances = sorted(relate.data, key=lambda x: x.get_metas())

    inst_pairs = zip(attach_instances, relate_instances)
    for i, (att, rel) in enumerate(inst_pairs):
        if not _instance_check(phrasebook, att, rel):
            print("mismatch of attachment/relation instance, "
                  "instance number", i,
                  _instance_help(phrasebook, att),
                  _instance_help(phrasebook, rel),
                  file=sys.stderr)
        prob, best = _combine_single_prob(attach, relate, att, rel)
        edu1, edu2 = edu_pair(att)
        distrib.append((edu1, edu2, prob, best))
    return distrib


def _add_labels(phrasebook, predicted, relate):
    """ predict labels for a given set of edges (=post-labelling an unlabelled
    decoding)

    :type relate: DataAndModel
    """
    rels = index_by_metas(relate.data,
                          metas=[phrasebook.source, phrasebook.target])
    result = []
    for edu1, edu2, _ in predicted:
        instance_rel = rels[(edu1, edu2)]
        rel = relate.model(instance_rel, Classifier.GetValue)
        result.append((edu1, edu2, rel))
    return result


def _instance_check(phrasebook, one, two):
    """
    Return True if the two annotations should be considered as refering to the
    same EDU pair. This can be used as a sanity check when zipping two datasets
    that are expected to be on the same EDU pairs.
    """
    return\
        one[phrasebook.source] == two[phrasebook.source] and\
        one[phrasebook.target] == two[phrasebook.target] and\
        one[phrasebook.grouping] == two[phrasebook.grouping]


def _instance_help(phrasebook, instance):
    """
    A hopefully small string that helps users to easily identify
    an instance at a glance when something goes wrong
    """
    return "%s: %s-%s" % (instance[phrasebook.grouping],
                          instance[phrasebook.source],
                          instance[phrasebook.target])


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


# pylint: disable=unused-argument
def _get_attach_prob_perceptron(config, attach):
    """
    Attachment probabilities (only) for each EDU pair in the data
    """
    if not attach.model.use_prob:
        raise DecoderException("ERROR: Trying to output probabilities, "
                               "while Perceptron parametrized with "
                               "use_prob=False!")
    return attach.model.get_scores(attach.data)
# pylint: enable=unused-argument

def _get_attach_prob_orange(config, attach):
    """
    Attachment probabilities (only) for each EDU pair in the data
    """
    # orange-based models
    edu_pair = mk_edu_pairs(config.phrasebook, attach.data.domain)
    prob_distrib = []
    for inst in attach.data:
        edu1, edu2 = edu_pair(inst)
        prob = _get_inst_attach_orange(attach.model, inst)
        prob_distrib.append((edu1, edu2, prob, "unlabelled"))
    return prob_distrib


def decode(config, decoder, attach, relate=None, nbest=1):
    """
    Decode every instance in the attachment table (predicting
    relations too if we have the data/model for it).
    Return the predictions made

    :type attach: DataAndModel
    :type relate: DataAndModel
    """

    # TODO issue #9: check that call to learner can be uniform
    # with 2 parameters (as logistic), as the documentation is
    # inconsistent on this
    if relate is not None and not config.post_labelling:
        prob_distrib = _combine_probs(config.phrasebook,
                                      attach, relate)
    elif is_perceptron_model(attach.model):
        # home-made online models
        prob_distrib = _get_attach_prob_perceptron(config, attach)
    else:
        # orange-based models
        prob_distrib = _get_attach_prob_orange(config, attach)
    # print prob_distrib

    # get prediction (input is just prob_distrib)
    # not all decoders support the threshold keyword argument
    # hence the apparent redundancy here
    # TODO: issue #8: PM CHECK if works with nbest decoding
    if config.threshold is not None:
        predicted = decoder(prob_distrib,
                            threshold=config.threshold,
                            use_prob=config.use_prob)
    else:
        predicted = decoder(prob_distrib,
                            use_prob=config.use_prob)

    if config.post_labelling:
        if nbest == 1:
            predicted = _add_labels(config.phrasebook, predicted, relate)
        else:
            predicted = [_add_labels(config.phrasebook, x, relate)
                         for x in predicted]

    return predicted


def count_correct(phrasebook,
                  predicted, reference,
                  labels=None, debug=False):
    """basic eval: counting correct predicted edges (labelled or not)
    data contains the reference attachments
    labels the corresponding relations

    return_type: `Count` object
    """
    score_attach = 0
    score_label = 0
    dict_predicted = {(arg1, arg2): rel for arg1, arg2, rel in predicted}
    for one in reference:
        edu_pair = (one[phrasebook.source].value,
                    one[phrasebook.target].value)
        if debug:
            print(edu_pair, dict_predicted.get(edu_pair), file=sys.stderr)
        if edu_pair in dict_predicted:
            score_attach += 1
            if labels is not None:
                relation_row = select_edu_pair(phrasebook, edu_pair, labels)
                if relation_row is not None:
                    relation_ref = relation_row[phrasebook.label].value
                    if dict_predicted[edu_pair] == relation_ref:
                        score_label += 1
                else:
                    print("attached pair without corresponding relation",
                          one[phrasebook.grouping], edu_pair,
                          file=sys.stderr)

    return Count(score_attach, score_label, len(predicted), len(reference))
