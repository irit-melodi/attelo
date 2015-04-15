'''
Oracles: return probabilities and values directly from gold data
'''

# Author: Eric Kow
# License: CeCILL-B (French BSD3)

from scipy.sparse import dok_matrix

from attelo.table import (UNRELATED,
                          UNKNOWN,
                          for_labelling)
from .interface import (LabelClassifier)


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
        dpack = for_labelling(dpack)
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
