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


class SklearnClassifier(object):
    """
    An sklearn classifier used for any purpose
    """
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
        best_idxes = np.argsort(weights)[-top_n:][::-1]
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


class SklearnAttachClassifier(AttachClassifier, SklearnClassifier):
    """A relatively simple way to get an attachment classifier:
    just pass in an sklearn classifier.

    Parameters
    ----------
    learner: sklearn API-compatible classifier
        The learner to use for label prediction.

    pos_label: str or int, 1 by default
        The class that codes an attachment decision.
    """

    def __init__(self, learner, pos_label=1):
        AttachClassifier.__init__(self)
        SklearnClassifier.__init__(self, learner)
        self._fitted = False
        self.pos_label = pos_label

    def fit(self, dpacks, targets, nonfixed_pairs=None):
        # WIP select only the nonfixed pairs
        if nonfixed_pairs is not None:
            dpacks = [dpack.selected(nf_pairs)
                      for dpack, nf_pairs in zip(dpacks, nonfixed_pairs)]
            targets = [target[nf_pairs]
                       for target, nf_pairs in zip(targets, nonfixed_pairs)]

        dpack = DataPack.vstack(dpacks)
        target = np.concatenate(targets)
        self._learner.fit(dpack.data, target)
        self._fitted = True
        return self

    def predict_score(self, dpack, nonfixed_pairs=None):
        if not self._fitted:
            raise ValueError('Fit not yet called')

        # WIP pass only nonfixed pairs to the classifier
        if nonfixed_pairs is not None:
            dpack_filtd = dpack.selected(nonfixed_pairs)
        else:
            dpack_filtd = dpack

        if self.can_predict_proba:
            attach_idx = list(self._learner.classes_).index(self.pos_label)
            probs = self._learner.predict_proba(dpack_filtd.data)
            scores_pred = probs[:, attach_idx]
        else:
            scores_pred = self._learner.decision_function(dpack_filtd.data)

        # WIP overwrite only the attachment scores of non-fixed pairs
        if nonfixed_pairs is not None:
            scores = np.copy(dpack.graph.attach)
            scores[nonfixed_pairs] = scores_pred
        else:
            scores = scores_pred

        return scores


class SklearnLabelClassifier(LabelClassifier, SklearnClassifier):
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
        SklearnClassifier.__init__(self, learner)
        self._fitted = False
        self._labels = None  # not yet learned

    def fit(self, dpacks, targets, nonfixed_pairs=None):
        # WIP select only the nonfixed pairs
        if nonfixed_pairs is not None:
            dpacks = [dpack.selected(nf_pairs)
                      for dpack, nf_pairs in zip(dpacks, nonfixed_pairs)]
            targets = [target[nf_pairs]
                       for target, nf_pairs in zip(targets, nonfixed_pairs)]

        dpack = DataPack.vstack(dpacks)
        target = np.concatenate(targets)
        self._learner.fit(dpack.data, target)
        self._labels = [dpack.get_label(x) for x in self._learner.classes_]
        self._fitted = True
        return self

    def predict_score(self, dpack, nonfixed_pairs=None):
        if not self._fitted:
            raise ValueError('Fit not yet called')

        if self._labels is None:
            raise ValueError('No labels associated with this classifier')

        # WIP don't pass the fixed pairs to the classifier
        if nonfixed_pairs is not None:
            dpack_filtd = dpack.selected(nonfixed_pairs)
        else:
            dpack_filtd = dpack

        # TODO non-probabilistic labellers
        weights = self._learner.predict_proba(dpack_filtd.data)
        lbl_scores_pred = relabel(self._labels, weights, dpack_filtd.labels)

        # WIP overwrite only the labelling scores of non-fixed pairs
        if nonfixed_pairs is not None:
            lbl_scores = np.copy(dpack.graph.label)
            lbl_scores[nonfixed_pairs] = lbl_scores_pred
        else:
            lbl_scores = lbl_scores_pred

        return lbl_scores
