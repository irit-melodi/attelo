"""
Baseline decoders
"""

from .util import get_sorted_edus, DecoderException


def local_baseline(prob_distrib, threshold = 0.5, use_prob=True):
    """just attach locally if prob is > threshold
    """
    predicted = []
    for (arg1, arg2, probs, label) in prob_distrib:
        attach = probs
        if use_prob:
            if attach > threshold:
                predicted.append((arg1.id, arg2.id, label))
        else:
            if attach >= 0.0:
                predicted.append((arg1.id, arg2.id, label))
    return predicted


def last_baseline(prob_distrib, use_prob=True):
    "attach to last, always"

    labels = {(edu1, edu2): lab for edu1, edu2, _, lab in prob_distrib}

    def get_prediction(edu1, edu2):
        "Return triple of EDU ids and label from the probability distribution"

        span1 = edu1.span()
        span2 = edu2.span()
        if (edu1, edu2) in labels:
            label = labels[(edu1, edu2)]
            return (edu1.id, edu2.id, label)
        elif span1 == span2:
            return None
        else:
            raise DecoderException("Could not find row with EDU pairs "
                                   "%s and %s: " % (edu1.id, edu2.id))

    edus = get_sorted_edus(prob_distrib)
    ordered_pairs = zip(edus[:-1], edus[1:])
    results = []
    for edu1, edu2 in ordered_pairs:
        prediction = get_prediction(edu1, edu2)
        if prediction:
            results.append(prediction)
    return results
