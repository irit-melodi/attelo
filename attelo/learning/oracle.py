'''
Oracles: return probabilities and values directly from gold data
'''

# Author: Eric Kow
# License: CeCILL-B (French BSD3)

# pylint: disable=no-name-in-module
import numpy as np
# pylint: enable=no-name-in-module

from scipy.sparse import dok_matrix

from attelo.table import (UNRELATED,
                          UNKNOWN)
from .interface import (AttachClassifier,
                        LabelClassifier)


class AttachOracle(AttachClassifier):
    """A faux attachment classifier that returns "probabilities"
    1.0 for gold attached links and 0.0 for gold unattached.

    Naturally, this would only do something useful if the
    test datapack comes with gold predictions.
    """
    def __init__(self):
        super(AttachOracle, self).__init__()
        self.can_predict_proba = True

    def fit(self, dpacks, targets, nonfixed_pairs=None):
        return self

    def predict_score(self, dpack, nonfixed_pairs=None):
        """Predict 1.0 for gold attachments, 0.0 otherwise.

        Notes
        -----
        This assumes that gold attachments are coded as 1 ; this assumption
        is currently baked in `attelo.table.for_attachment`.

        TODO
        ----
        [ ] rename and refactor to predict_proba(self, dpacks)
        """
        if nonfixed_pairs is not None:
            y_true = dpack.target[nonfixed_pairs]
        else:
            y_true = dpack.target

        score_true = np.where(y_true == 1, 1.0, 0.0)

        if nonfixed_pairs is not None:
            res = np.copy(dpack.graph.attach)
            res[nonfixed_pairs] = score_true
        else:
            res = score_true

        return res


class LabelOracle(LabelClassifier):
    '''
    A faux label classifier that returns "probabilities"
    1.0 for the gold label and 0.0 other labels.

    Note: edges which are unrelated in the gold will be
    assigned the special "unknown" label (something like
    '__UNK__')

    Naturally, this would only do something useful if the
    datapack comes with gold labels
    '''
    def __init__(self):
        super(LabelOracle, self).__init__()
        self.can_predict_proba = True

    def fit(self, dpacks, targets, nonfixed_pairs=None):
        return self

    def predict_score(self, dpack, nonfixed_pairs=None):
        """Predict 1.0 for the gold label of edges.

        Non-gold edges are attributed "unknown" for their gold label.

        Parameters
        ----------
        dpack : DataPack
            Datapack for which we want labelling scores.

        Returns
        -------
        scores : 2D dense array of float
            Scores for each pairing and each label.

        TODO
        ----
        [ ] rename and refactor to predict_proba(self, dpacks)
        """
        weights = dok_matrix((len(dpack), len(dpack.labels)))
        lbl_unrelated = dpack.label_number(UNRELATED)
        lbl_unk = dpack.label_number(UNKNOWN)

        if nonfixed_pairs is not None:
            for i, lbl in enumerate(dpack.target):
                if i in nonfixed_pairs:
                    if lbl == lbl_unrelated:
                        weights[i, lbl_unk] = 1.0
                    else:
                        weights[i, lbl] = 1.0
                else:
                    weights[i, :] = dpack.graph.label[i]
        else:
            for i, lbl in enumerate(dpack.target):
                if lbl == lbl_unrelated:
                    weights[i, lbl_unk] = 1.0
                else:
                    weights[i, lbl] = 1.0

        return weights.todense()
