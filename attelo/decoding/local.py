"""
Local decoders make decisions for each edge independently.
"""

from .interface import Decoder
from .util import (convert_prediction,
                   simple_candidates)


class AsManyDecoder(Decoder):
    """Greedy decoder that picks as many edges as there are real EDUs.

    The output structure is a graph that has the same number of edges
    as a spanning tree over the EDUs.
    It can be non-connex, contain cycles and re-entrancies.
    """

    def decode(self, dpack):
        """Return the set of top N edges
        """
        cands = simple_candidates(dpack)
        # number of real EDUs
        nb_edus = len(dpack.edus)
        # sort candidates by their scores (in reverse order)
        sorted_cands = sorted(cands, key=lambda c: c[2], reverse=True)
        # take the top N candidates, where N is the number of real EDUs
        predicted = [(src.id, tgt.id, lbl)
                     for src, tgt, _, lbl in sorted_cands[:nb_edus]]
        return convert_prediction(dpack, predicted)


class BestIncomingDecoder(Decoder):
    """Greedy decoder that picks the best incoming edge for each EDU.

    The output structure is a graph that contains exactly one incoming
    edge for each EDU, thus it has the same number of edges as a
    spanning tree over the EDUs.
    It can be non-connex or contain cycles, but no re-entrancy.
    """

    def decode(self, dpack):
        """Return the best incoming edge for each EDU
        """
        # best incoming edge for each EDU
        inc_edges = {}
        for cand in simple_candidates(dpack):
            src, tgt, score, lbl = cand
            if tgt.id in inc_edges:
                cur_best = inc_edges[tgt.id][2]
                if score > cur_best:
                    inc_edges[tgt.id] = cand
            else:
                inc_edges[tgt.id] = cand

        predicted = [(src.id, tgt.id, lbl)
                     for src, tgt, score, lbl in inc_edges.values()]
        return convert_prediction(dpack, predicted)
