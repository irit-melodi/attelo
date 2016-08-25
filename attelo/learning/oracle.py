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

    def fit(self, dpacks, targets):
        return self

    def predict_score(self, dpack):
        """Predict 1.0 for gold attachments, 0.0 otherwise.

        Notes
        -----
        This assumes that gold attachments are coded as 1 ; this assumption
        is currently baked in `attelo.table.for_attachment`.

        TODO
        ----
        [ ] rename and refactor to predict_proba(self, dpacks)
        """
        y_true = dpack.target
        score_true = np.where(y_true == 1, 1.0, 0.0)
        return score_true


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

    def fit(self, dpacks, targets):
        return self

    def predict_score(self, dpack):
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

        y_true = dpack.target
        # for each pairing, if the true label is "unrelated", set 1.0
        # score to "unknown" instead (enables gold labelling on non-gold
        # attachment)
        lbl_unrelated = dpack.label_number(UNRELATED)
        lbl_unk = dpack.label_number(UNKNOWN)
        lbl_true = np.where(y_true != lbl_unrelated, y_true, lbl_unk)
        weights[np.arange(len(weights)), lbl_true] = 1.0

        return weights.todense()
