"""
Local classifiers
"""

# pylint: disable=no-name-in-module
from numpy import (concatenate as np_concatenate)
# pylint: enable=no-name-in-module

from attelo.table import (DataPack,
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
