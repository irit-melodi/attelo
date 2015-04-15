"""
Local classifiers
"""

from scipy.sparse import dok_matrix

from attelo.table import (DataPack,
                          UNRELATED,
                          UNKNOWN,
                          for_labelling)
from .interface import (LabelClassifier)


class SklearnLabelClassifier(LabelClassifier):
    '''
    A relative simple way to get a label classifier: just
    pass in a scikit classifier

    Attributes
    ----------
    labels: [string]
        (fitted) List of labels on which this classifier
        will emit scores
    '''

    def __init__(self, learner):
        """
        learner: scikit-compatible classifier
            Use the given learner for label prediction.
        """
        super(SklearnLabelClassifier, self).__init__()
        self._learner = learner
        pfunc = getattr(learner, "predict_proba", None)
        self.can_predict_proba = callable(pfunc)
        self._fitted = False
        self.labels = None  # not yet learned

    def fit(self, mpack):
        mpack = {k: for_labelling(v.attached_only())
                 for k, v in mpack.items()}
        dpack = DataPack.vstack(mpack.values())
        self._learner.fit(dpack.data, dpack.target)
        self.labels = [dpack.get_label(x) for x in self._learner.classes_]
        self._fitted = True
        return self

    def transform(self, dpack):
        dpack = for_labelling(dpack)
        if self.labels is None:
            raise ValueError('No labels associated with this classifier')
        if not self._fitted:
            raise ValueError('Fit not yet called')
        return self.labels, self._learner.predict_proba(dpack.data)


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

    def fit(self, _):
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
