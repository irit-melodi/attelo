"""
Local classifiers
"""

from __future__ import print_function

import numpy as np

from attelo.cdu import CDU
from attelo.table import DataPack
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

    def fit(self, dpacks, targets):
        dpack = DataPack.vstack(dpacks)
        target = np.concatenate(targets)
        self._learner.fit(dpack.data, target)
        self._fitted = True
        return self

    def predict_score(self, dpack):
        if not self._fitted:
            raise ValueError('Fit not yet called')

        if self.can_predict_proba:
            attach_idx = list(self._learner.classes_).index(self.pos_label)
            probs = self._learner.predict_proba(dpack.data)
            scores_pred = probs[:, attach_idx]
        else:
            scores_pred = self._learner.decision_function(dpack.data)
        if False:
            # scoring of CDU pairings is currently de-activated
            scores_pred = self.overwrite_scores_cdu(dpack, scores_pred)
        return scores_pred

    def overwrite_scores_cdu(self, dpack, scores_pred):
        """Overwrite scores of EDU pairings with scores of CDU pairings.

        Parameters
        ----------
        dpack : DataPack
            DataPack

        scores_pred : array of float, dimensions=(len(edu_pairings), 1)
            Predicted scores for pairs of EDUs.

        Returns
        -------
        scores_pred : array of float, dimensions=(len(edu_pairings), 1)
            Updated array of predicted scores.

        Notes
        -----
        All CDU-related code might be better off in a dedicated submodule
        of parser.
        """
        attach_idx = list(self._learner.classes_).index(self.pos_label)
        # 2016-07-29 WIP compute scores for CDUs
        # TODO find a cleaner way to compute these scores
        epairs_map = {(src.id, tgt.id): i for i, (src, tgt)
                      in enumerate(dpack.pairings)}
        # filter CDU pairs
        cpairs_idc = []  # indices of selected CDU pairs
        epairs_idc = []  # indices of corresponding EDU pairs
        for i, (src, tgt) in enumerate(dpack.cdu_pairings):
            esrc = src.members[0] if isinstance(src, CDU) else src.id
            etgt = tgt.members[0] if isinstance(tgt, CDU) else tgt.id
            if (esrc, etgt) in epairs_map:
                cpairs_idc.append(i)
                epairs_idc.append(epairs_map[(esrc, etgt)])
        # score pairs on CDUs, replace the score of the EDU pair if the
        # score of the CDU pair is higher
        if cpairs_idc:
            print('woot!')  # DEBUG
            sel_cdu_data = dpack.cdu_data[cpairs_idc]
            if self.can_predict_proba:
                cdu_probs = self._learner.predict_proba(sel_cdu_data)
                cdu_scores_pred = cdu_probs[:, attach_idx]
            else:
                cdu_scores_pred = self._learner.decision_function(
                    sel_cdu_data)
            # DEBUG
            if False:
                epairs = [dpack.pairings[i] for i in epairs_idc]
                epair_ids = [(src.id, tgt.id) for src, tgt in epairs]
                print('epair scores')
                for x, y in zip(epair_ids,
                                [scores_pred[i] for i in epairs_idc])[:30]:
                    print(x, y)
                cpairs = [dpack.cdu_pairings[i] for i in cpairs_idc]
                cpair_ids = [(src if isinstance(src, CDU) else src.id,
                              tgt if isinstance(tgt, CDU) else tgt.id)
                             for src, tgt in cpairs]
                print('cpair scores')
                for x, y in zip(cpair_ids, cdu_scores_pred)[:30]:
                    print(x, y)
                raise ValueError('gne')
            # end DEBUG
            # was: np.maximum(scores_pred[epairs_idc], cdu_scores_pred)
            scores_pred[epairs_idc] = cdu_scores_pred
        # end WIP CDUs

        return scores_pred


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

    def fit(self, dpacks, targets):
        dpack = DataPack.vstack(dpacks)
        target = np.concatenate(targets)
        self._learner.fit(dpack.data, target)
        self._labels = [dpack.get_label(x) for x in self._learner.classes_]
        self._fitted = True
        return self

    def predict_score(self, dpack):
        if not self._fitted:
            raise ValueError('Fit not yet called')

        if self._labels is None:
            raise ValueError('No labels associated with this classifier')

        # TODO non-probabilistic labellers
        weights = self._learner.predict_proba(dpack.data)
        lbl_scores_pred = relabel(self._labels, weights, dpack.labels)

        return lbl_scores_pred
