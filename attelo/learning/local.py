"""
Local classifiers
"""

import numpy as np

from attelo.table import (DataPack,
                          for_labelling)
from .interface import (AttachClassifier,
                        LabelClassifier)
from .util import (relabel)

class SklearnAttachClassifier(AttachClassifier):
    '''
    A relatively simple way to get an attachment classifier:
    just pass in a scikit classifier
    '''

    def __init__(self, learner):
        """
        learner: scikit-compatible classifier
            Use the given learner for label prediction.
        """
        super(SklearnAttachClassifier, self).__init__()
        self._learner = learner
        pfunc = getattr(learner, "predict_proba", None)
        self.can_predict_proba = callable(pfunc)
        self._fitted = False

    def fit(self, dpacks, targets):
        dpack = DataPack.vstack(dpacks)
        target = np.concatenate(targets)
        self._learner.fit(dpack.data, target)
        self._fitted = True
        return self

    def transform(self, dpack):
        if not self._fitted:
            raise ValueError('Fit not yet called')
        elif self.can_predict_proba:
            attach_idx = list(self._learner.classes_).index(1)
            probs = self._learner.predict_proba(dpack.data)
            return probs[:, attach_idx]
        else:
            return self._learner.decision_function(dpack.data)


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
        self._labels = None  # not yet learned

    def fit(self, dpacks, targets):
        dpack = DataPack.vstack(dpacks)
        target = np.concatenate(targets)
        self._learner.fit(dpack.data, target)
        self._labels = [dpack.get_label(x) for x in self._learner.classes_]
        self._fitted = True
        return self

    def transform(self, dpack):
        dpack, _ = for_labelling(dpack, dpack.target)
        if self._labels is None:
            raise ValueError('No labels associated with this classifier')
        if not self._fitted:
            raise ValueError('Fit not yet called')
        weights = self._learner.predict_proba(dpack.data)
        return relabel(self._labels, weights, dpack.labels)
