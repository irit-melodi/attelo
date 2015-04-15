"""
Local classifiers
"""

from scipy.sparse import dok_matrix
# pylint: disable=no-name-in-module
from numpy import (concatenate as np_concatenate,
                   take as np_take,
                   where as np_where)
# pylint: enable=no-name-in-module

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

    def fit(self, dpacks, targets):
        dpack = DataPack.vstack(dpacks)
        target = np_concatenate(targets)
        # select attached, apply label filter
        # we can't just use dpack.attached_only, because we need
        # to retrieve the indices to select the targets with
        unrelated = dpack.label_number(UNRELATED)
        indices = np_where(dpack.target != unrelated)[0]
        dpack = for_labelling(dpack.selected(indices))
        target = np_take(target, indices)
        # now go
        self._learner.fit(dpack.data, target)
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
