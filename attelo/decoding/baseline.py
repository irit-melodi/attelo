"""
Baseline decoders
"""

from .interface import Decoder
from .util import get_sorted_edus, get_prob_map, DecoderException

# pylint: disable=too-few-public-methods


class LocalBaseline(Decoder):
    """just attach locally if prob is > threshold"""
    def __init__(self, threshold, use_prob=True):
        self._threshold = threshold
        self._use_prob = use_prob

    def __call__(self, prob_distrib):
        predicted = []
        for arg1, arg2, probs, label in prob_distrib:
            attach = probs
            threshold = self._threshold if self._use_prob else 0.0
            if attach > threshold:
                predicted.append((arg1.id, arg2.id, label))
        return [predicted]


class LastBaseline(Decoder):
    "attach to last, always"

    def __call__(self, prob_distrib):
        labels_probs = get_prob_map(prob_distrib)

        def get_prediction(edu1, edu2):
            """Return triple of EDU ids and label from the probability
            distribution (or None if no attachment)"""
            span1 = edu1.span()
            span2 = edu2.span()
            if (edu1, edu2) in labels_probs:
                label, _ = labels_probs[(edu1, edu2)]
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
        return [results]
