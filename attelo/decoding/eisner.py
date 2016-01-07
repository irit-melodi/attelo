"""Eisner decoder
"""

import numpy as np

from .interface import Decoder
# temporary? imports
from ..table import _edu_positions
from .util import (cap_score, convert_prediction, simple_candidates,
                   MIN_SCORE)


class EisnerDecoder(Decoder):
    """The Eisner decoder builds projective dependency trees.

    Parameters
    ----------
    use_prob: boolean, optional
        If True, the scores retrieved from the model are considered as
        probabilities and projected from [0,1] to ]-inf, 0] using the
        log function.
        Defaults to True.

    unique_real_root: boolean, optional
        If True, each output tree will have a unique real root, i.e. the
        fake root node will have a unique child.
        Defaults to True.
    """

    def __init__(self, unique_real_root=True, use_prob=True):
        self._unique_real_root = unique_real_root
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
        # whether the output tree should contain a unique real root
        unique_real_root = self._unique_real_root

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
                 (cap_score(np.log(attach_score)) if self._use_prob
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
        cscores = np.zeros((nb_edus, nb_edus, 2, 2), dtype=np.float64)
        # backpointers: index of split point
        csplits = np.zeros((nb_edus, nb_edus, 2, 2), dtype=np.int32)

        # iterate over all possible spans of increasing size
        for span in range(1, nb_edus):
            for start in range(nb_edus - span):
                end = start + span

                # left open
                range_k = range(start, end)
                # find argmax and max on range_k
                cands = [(cscores[start][k][0][1] +
                          cscores[k + 1][end][1][1] +
                          (score[(end, start)]
                           if start > 0 and (end, start) in score
                           else MIN_SCORE))
                         for k in range_k]
                max_cand = np.nanmax(cands)
                argmax_cand = (range_k[cands.index(max_cand)]
                               if not np.isnan(max_cand)
                               else range_k[0])
                # update tables
                cscores[start][end][1][0] = max_cand
                csplits[start][end][1][0] = argmax_cand

                # right open
                # if start == 0, restricting range_k to [0]
                # enforces that the tree has a unique real root
                range_k = ([0] if unique_real_root and start == 0
                           else range(start, end))
                # find argmax and max on range_k
                cands = [(cscores[start][k][0][1] +
                          cscores[k + 1][end][1][1] +
                          (score[(start, end)]
                           if (start, end) in score
                           else MIN_SCORE))
                         for k in range_k]
                max_cand = np.nanmax(cands)
                argmax_cand = (range_k[cands.index(max_cand)]
                               if not np.isnan(max_cand)
                               else range_k[0])
                # update tables
                cscores[start][end][0][0] = max_cand
                csplits[start][end][0][0] = argmax_cand

                # left closed
                range_k = range(start, end)
                # find argmax and max on range_k
                cands = [(cscores[start][k][1][1] +
                          cscores[k][end][1][0])
                         for k in range_k]
                max_cand = np.nanmax(cands)
                argmax_cand = (range_k[cands.index(max_cand)]
                               if not np.isnan(max_cand)
                               else range_k[0])
                # update tables
                cscores[start][end][1][1] = max_cand
                csplits[start][end][1][1] = argmax_cand

                # right closed
                range_k = range(start + 1, end + 1)
                # find argmax and max on range_k
                cands = [(cscores[start][k][0][0] +
                          cscores[k][end][0][1])
                         for k in range_k]
                max_cand = np.nanmax(cands)
                argmax_cand = (range_k[cands.index(max_cand)]
                               if not np.isnan(max_cand)
                               else range_k[0])
                # update tables
                cscores[start][end][0][1] = max_cand
                csplits[start][end][0][1] = argmax_cand

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
