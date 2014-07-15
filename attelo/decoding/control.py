"""
Central interface to the decoders
"""

from __future__ import print_function
from Orange.classification import Classifier
import sys

from attelo.edu import mk_edu_pairs
from attelo.table import index_by_metas


# TODO: replace with named tuple and constructor
class DecoderConfig(object):
    def __init__(self, features, decoder,
                 threshold=None,
                 post_labelling=False,
                 use_prob=True):
        self.features = features
        self.decoder = decoder
        self.threshold = threshold
        self.post_labelling = post_labelling
        self.use_prob = use_prob

# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


def _combine_probs(features,
                   attach_instances,
                   rel_instances,
                   attachmt_model,
                   relations_model):
    """retrieve probability of the best relation on an edu pair, given the
    probability of an attachment
    """
    # !! instances set must correspond to same edu pair in the same order !!
    distrib = []

    edu_pair = mk_edu_pairs(features, attach_instances.domain)
    attach_instances = sorted(attach_instances, key=lambda x: x.get_metas())
    rel_instances = sorted(rel_instances, key=lambda x: x.get_metas())

    inst_pairs = zip(attach_instances, rel_instances)
    for i, (attach, relation) in enumerate(inst_pairs):
        p_attach = attachmt_model(attach, Classifier.GetProbabilities)[1]
        p_relations = relations_model(relation, Classifier.GetBoth)
        if not _instance_check(features, attach, relation):
            print("mismatch of attachment/relation instance, "
                  "instance number", i,
                  meta_info(attach, features),
                  meta_info(relation, features),
                  file=sys.stderr)
        # this should be investigated
        try:
            best_rel = p_relations[0].value
        except:
            best_rel = p_relations[0]

        rel_prob = max(p_relations[1])
        edu1, edu2 = edu_pair(attach)
        distrib.append((edu1, edu2, p_attach * rel_prob, best_rel))
    return distrib


def _add_labels(features, predicted, rel_instances, relations_model):
    """ predict labels for a given set of edges (=post-labelling an unlabelled
    decoding)
    """
    rels = index_by_metas(rel_instances,
                          metas=[features.source, features.target])
    result = []
    for (a1, a2, _r) in predicted:
        instance_rel = rels[(a1, a2)]
        rel = relations_model(instance_rel,
                              Classifier.GetValue)
        result.append((a1, a2, rel))
    return result


def _instance_check(features, one, two):
    """
    Return True if the two annotations should be considered as refering to the
    same EDU pair. This can be used as a sanity check when zipping two datasets
    that are expected to be on the same EDU pairs.
    """
    return\
        one[features.source] == two[features.source] and\
        one[features.target] == two[features.target] and\
        one[features.grouping] == two[features.grouping]


def meta_info(features, instance):
    return "%s: %s-%s" % (instance[features.grouping],
                          instance[features.source],
                          instance[features.target])


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


def decode_document(config,
                    model_attach, attach_instances,
                    model_relations=None, rel_instances=None):
    """
    decode one document (onedoc), selecting instances for attachment from
    data_attach, (idem relations if present), using trained model, model

    Return the predictions made

    TODO: check that call to learner can be uniform with 2 parameters (as
    logistic), as the documentation is inconsistent on this
    """
    features = config.features
    decoder = config.decoder
    threshold = config.threshold
    use_prob = config.use_prob

    if rel_instances and not config.post_labelling:
        prob_distrib = _combine_probs(features,
                                      attach_instances, rel_instances,
                                      model_attach, model_relations)
    elif model_attach.name in ["Perceptron", "StructuredPerceptron"]:
        # home-made online models
        prob_distrib = model_attach.get_scores(attach_instances,
                                               use_prob=use_prob)
    else:
        # orange-based models
        edu_pair = mk_edu_pairs(features, attach_instances.domain)
        prob_distrib = []
        for one in attach_instances:
            edu1, edu2 = edu_pair(one)
            probs = model_attach(one, Classifier.GetProbabilities)[1]
            prob_distrib.append((edu1, edu2, probs, "unlabelled"))
    # print prob_distrib

    # get prediction (input is just prob_distrib)
    # not all decoders support the threshold keyword argument
    # hence the apparent redundancy here
    if threshold:
        predicted = decoder(prob_distrib,
                            threshold=threshold,
                            use_prob=use_prob)
        # predicted = decoder(prob_distrib, threshold = threshold)
    else:
        predicted = decoder(prob_distrib,
                            use_prob=use_prob)
        # predicted = decoder(prob_distrib)

    if config.post_labelling:
        predicted = _add_labels(features,
                                predicted,
                                rel_instances,
                                model_relations)
        # predicted = _add_labels(predicted, rel_instances, model_relations)

    return predicted
