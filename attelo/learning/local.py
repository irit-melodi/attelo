"""
Local classifiers
"""

import numpy as np

from attelo.table import (DataPack,
                          for_labelling)
from .interface import (AttachClassifier,
                        LabelClassifier)
from .util import (relabel)

# pylint: disable=too-few-public-methods


class SkClassifier(object):
    '''
    An scikit classifier used for any purpose
    '''
    def __init__(self, learner):
        self._learner = learner
        pfunc = getattr(learner, "predict_proba", None)
        self.can_predict_proba = callable(pfunc)

    @staticmethod
    def _best_weights(weights, top_n):
        """
        Given an array of weights, return the `top_n` most important
        ones and the associated weight

        Return
        ------
        best: [(int, float)]
            index within the input weight array, and score
        """
        best_idxes = np.argsort(np.absolute(weights))[-top_n:][::-1]
        best_weights = np.take(weights, best_idxes)
        return zip(best_idxes, best_weights)

    def important_features(self, top_n):
        """
        If possible, return a list of important features with
        their weights.

        The underlying classifier must provide either a `coef_`
        or `feature_importances_` property

        Return
        ------
        features: None or [(int, float)]
            the features themselves are indices into some vocabulary
        """
        model = self._learner
        if hasattr(model, 'coef_') and len(model.classes_) <= 2:
            return self._best_weights(model.coef_[0], top_n)
        elif hasattr(model, 'feature_importances_'):
            return self._best_weights(model.feature_importances_, top_n)
        else:
            return None

    def important_features_multi(self, top_n):
        """
        If possible, return a dictionary mapping class indices
        to important features

        The underlying classifier must provide a `coef_` property

        Return
        ------
        feature_map: None or dict(int, [(int, float)])
            keys are label (indices) as you would find in a
            datapack; features are indices into a vocabulary
        """
        model = self._learner
        if hasattr(model, 'coef_'):
            res = {}
            for i, lbl in enumerate(model.classes_):
                res[lbl] = self._best_weights(model.coef_[i], top_n)
            return res
        else:
            return None


class SklearnAttachClassifier(AttachClassifier, SkClassifier):
    '''
    A relatively simple way to get an attachment classifier:
    just pass in a scikit classifier
    '''

    def __init__(self, learner):
        """
        learner: scikit-compatible classifier
            Use the given learner for label prediction.
        """
        AttachClassifier.__init__(self)
        SkClassifier.__init__(self, learner)
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


class SklearnLabelClassifier(LabelClassifier, SkClassifier):
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
        LabelClassifier.__init__(self)
        SkClassifier.__init__(self, learner)
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
