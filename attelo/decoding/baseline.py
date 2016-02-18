"""
Baseline decoders
"""

from .interface import Decoder
from .util import (convert_prediction,
                   get_sorted_edus,
                   get_prob_map,
                   DecoderException,
                   simple_candidates)

# pylint: disable=too-few-public-methods


class LocalBaseline(Decoder):
    """just attach locally if prob is > threshold"""
    def __init__(self, threshold, use_prob=True):
        self._threshold = threshold if use_prob else 0.0

    def decode(self, dpack, nonfixed_pairs=None):
        # TODO integrate nonfixed_pairs, maybe?
        cands = simple_candidates(dpack)
        results = [(e1.id, e2.id, lab) for e1, e2, w, lab in cands
                   if w > self._threshold]
        return convert_prediction(dpack, results)


class LastBaseline(Decoder):
    "attach to last, always"

    def decode(self, dpack, nonfixed_pairs=None):
        # TODO integrate nonfixed_pairs, maybe?
        cands = simple_candidates(dpack)
        labels_probs = get_prob_map(cands)

        def get_prediction(edu1, edu2):
            """Return triple of EDU ids and label from the probability
            distribution (or None if no attachment)"""
            span1 = edu1.span()
            span2 = edu2.span()
            id_pair = (edu1.id, edu2.id)
            if id_pair in labels_probs:
                label, _ = labels_probs[id_pair]
                return (edu1.id, edu2.id, label)
            elif span1 == span2:
                return None
            else:
                raise DecoderException("Could not find row with EDU pairs "
                                       "%s and %s: " % (edu1.id, edu2.id))

        edus = get_sorted_edus(cands)
        ordered_pairs = zip(edus[:-1], edus[1:])
        results = []
        for edu1, edu2 in ordered_pairs:
            prediction = get_prediction(edu1, edu2)
            if prediction:
                results.append(prediction)
        return convert_prediction(dpack, results)
