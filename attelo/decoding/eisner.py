"""Eisner decoder
"""

import numpy as np

from .interface import Decoder

# temporary? imports
from scipy.special import logit
from ..table import _edu_positions
from .mst import _cap_score
from .util import convert_prediction, simple_candidates


class EisnerDecoder(Decoder):
    """The Eisner decoder builds projective dependency trees.
    """

    def __init__(self, use_prob=True):
        self._use_prob = use_prob  # yerk

    def decode(self, dpack):
        """Decode

        Parameters
        ----------
        dpack: DataPack
            Datapack that describes the (sub)document to be parsed.

        Returns
        -------
        dpack_pred: DataPack
            A copy of the argument DataPack with predictions set.
        """
        # get number of EDUs and possible labels
        nb_edus = len(dpack.edus)
        # nb_lbls = dpack.graph.label.shape[0]

        # map EDU ids to their index in the document
        edu_id2idx = _edu_positions(dpack)
        # we will need the reverse mapping to produce the result
        edu_idx2id = {edu_idx: edu_id
                      for edu_id, edu_idx in edu_id2idx.items()}
        # build dict((src_idx, tgt_idx), (attach_score, lbl_scores))
        simple_cands = simple_candidates(dpack)
        # FIXME scores (probabilities or discriminative scores) should
        # be adapted before this point ;
        # should be (src, tgt): attach_score
        score = {(edu_id2idx[src.id], edu_id2idx[tgt.id]):
                 (_cap_score(logit(attach_score)) if self._use_prob
                  else attach_score)
                 for src, tgt, attach_score, best_lbl
                 in simple_cands}
        label = {(edu_id2idx[src.id], edu_id2idx[tgt.id]):
                 dpack.label_number(best_lbl)
                 for src, tgt, attach_score, best_lbl
                 in simple_cands}
        # end attelo-isms

        # Eisner algorithm
        # arrays of substructures for dynamic programming
        # [start][end][dir][complete]
        # scores
        cscores = np.zeros((nb_edus, nb_edus, 2, 2), dtype=np.float32)
        # index of split point (to backtrack)
        csplits = np.zeros((nb_edus, nb_edus, 2, 2), dtype=np.int32)
        # iterate over all possible spans of increasing size
        for span in range(1, nb_edus):
            for start in range(nb_edus - span):
                end = start + span

                # best "open": find pair of substructures with highest score
                split_scores = [(cscores[start][k][0][1] +
                                 cscores[k + 1][end][1][1])
                                for k in range(start, end)]
                max_split_score = max(split_scores)
                # then argmax to get the split point
                best_split_point = start + split_scores.index(max_split_score)
                # fill table for both directions of attachment
                # left attachment: impossible if start == fake root
                if start == 0 or (end, start) not in score:
                    cscores[start][end][1][0] = np.NINF
                else:
                    cscores[start][end][1][0] = (max_split_score +
                                                 score[(end, start)])
                    csplits[start][end][1][0] = best_split_point
                # right attachment
                if (start, end) not in score:
                    cscores[start][end][0][0] = np.NINF
                else:
                    cscores[start][end][0][0] = (max_split_score +
                                                 score[(start, end)])
                    csplits[start][end][0][0] = best_split_point

                # best "closed"
                # left attachment: impossible if start == fake root
                if start > 0:
                    split_scores = [(cscores[start][k][1][1] +
                                     cscores[k][end][1][0])
                                    for k in range(start, end)]
                    max_split_score = max(split_scores)
                    best_split_point = (start +
                                        split_scores.index(max_split_score))
                    cscores[start][end][1][1] = max_split_score
                    csplits[start][end][1][1] = best_split_point
                # right attachment
                split_scores = [(cscores[start][k][0][0] +
                                 cscores[k][end][0][1])
                                for k in range(start + 1, end + 1)]
                max_split_score = max(split_scores)
                best_split_point = (start + 1 +
                                    split_scores.index(max_split_score))
                cscores[start][end][0][1] = max_split_score
                csplits[start][end][0][1] = best_split_point


        # solution: C[0][n][->][1]
        # use the backpointers in csplits to get the best tree
        predictions = []
        backpointers = [(0, nb_edus - 1, 0, 1)]
        while backpointers:
            start, end, dir_la, complete = backpointers.pop()
            if start == end:
                continue
            k = csplits[start][end][dir_la][complete]
            if complete:
                # queue backpointers
                if dir_la:
                    backpointers.extend([(start, k, dir_la, 1),
                                         (k, end, dir_la, 0)])
                else:
                    backpointers.extend([(start, k, dir_la, 0),
                                         (k, end, dir_la, 1)])
            else:
                # add the underlying edge to the set of predictions
                if dir_la:
                    predictions.append((end, start, label[(end, start)]))
                else:
                    predictions.append((start, end, label[(start, end)]))
                # queue backpointers
                backpointers.extend([(start, k, 0, 1),
                                     (k + 1, end, 1, 1)])

        # resume attelo-isms
        # transform predictions to the expected format
        # back to EDU ids and relation labels as strings
        att_preds = [(edu_idx2id[src], edu_idx2id[tgt], dpack.get_label(lbl))
                     for src, tgt, lbl in predictions]
        # and integrate predictions into the datapack
        dpack_pred = convert_prediction(dpack, att_preds)

        return dpack_pred
