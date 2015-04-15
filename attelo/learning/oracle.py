'''
Oracles: return probabilities and values directly from gold data
'''

# Author: Eric Kow
# License: CeCILL-B (French BSD3)

# pylint: disable=no-name-in-module
from numpy import (vectorize as np_vectorize)
# pylint: enable=no-name-in-module

from scipy.sparse import dok_matrix

from attelo.table import (UNRELATED,
                          UNKNOWN)
from .interface import (AttachClassifier,
                        LabelClassifier)


class AttachOracle(AttachClassifier):
    '''
    A faux attachment classifier that returns "probabilities"
    1.0 for gold attached links and 0.0 for gold unattached

    Naturally, this would only do something useful if the
    test datapack comes with gold predictions
    '''
    def __init__(self):
        super(AttachOracle, self).__init__()
        self.can_predict_proba = True

    def fit(self, dpacks, targets):
        return self

    def transform(self, dpack):
        # nb: this isn't actually faster with vectorize
        to_prob = lambda x: 1.0 if x == 1.0 else 0.0
        return np_vectorize(to_prob)(dpack.target)


class LabelOracle(LabelClassifier):
    '''
    A faux label classifier that returns "probabilities"
    1.0 for the gold label and 0.0 other labels.

    Note: edges which are unrelated in the gold will be
    assigned the a special "unknown" label (something like
    '__UNK__')

    Naturally, this would only do something useful if the
    datapack comes with gold labels
    '''
    def __init__(self):
        super(LabelOracle, self).__init__()
        self.can_predict_proba = True

    def fit(self, dpacks, targets):
        return self

    def transform(self, dpack):
        labels = [UNKNOWN] + dpack.labels
        scores_l = dok_matrix((len(dpack), len(labels)))
        lbl_unrelated = dpack.label_number(UNRELATED)
        lbl_unk = dpack.label_number(UNKNOWN)
        for i, lbl in enumerate(dpack.target):
            if lbl == lbl_unrelated:
                scores_l[i, lbl_unk] = 1.0
            else:
                scores_l[i, lbl] = 1.0
        scores_l = scores_l.todense()
        return labels, scores_l
