"""
Local decoders make decisions for each edge independently.
"""

from collections import defaultdict

from .interface import Decoder


class AsManyDecoder(Decoder):
    """Greedy decoder that picks as many edges as there are real EDUs.

    The output structure is a graph that has the same number of edges
    as a spanning tree over the EDUs.
    It can be non-connex, contain cycles and re-entrancies.
    """

    def decode(self, cands):
        """Return the set of top N edges

        Parameters
        ----------
        cands: iterable
            candidate set
        """
        # number of real EDUs
        # this works under the assumption that all real EDUs appear
        # as targets in cands
        nb_edus = len(set(tgt for src, tgt, score, lbl in cands))
        # sort candidates by their scores (in reverse order)
        sorted_cands = sorted(cands, key=lambda c: c[2], reverse=True)
        # take the top N candidates, where N is the number of real EDUs
        predicted = [(src.id, tgt.id, lbl)
                     for src, tgt, score, lbl in sorted_cands[:nb_edus]]

        return [predicted]


class BestIncomingDecoder(Decoder):
    """Greedy decoder that picks the best incoming edge for each EDU.
    
    The output structure is a graph that contains exactly one incoming
    edge for each EDU, thus it has the same number of edges as a
    spanning tree over the EDUs.
    It can be non-connex or contain cycles, but no re-entrancy.
    """

    def decode(self, cands):
        """Return the best incoming edge for each EDU

        Parameters
        ----------
        cands: iterable
            candidate set
        """
        # best incoming edge for each EDU
        inc_edges = {}
        for cand in cands:
            src, tgt, score, lbl = cand
            if tgt.id in inc_edges:
                cur_best = inc_edges[tgt.id][2]
                if score > cur_best:
                    inc_edges[tgt.id] = cand
            else:
                inc_edges[tgt.id] = cand

        predicted = [(src.id, tgt.id, lbl)
                     for src, tgt, score, lbl in inc_edges.values()]

        return [predicted]
    
