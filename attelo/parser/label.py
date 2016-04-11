"""
Labelling
"""

from os import path as fp

import joblib
import numpy as np

from .interface import Parser
from attelo.table import (UNKNOWN, attached_only, for_labelling,
                          idxes_attached)


class LabelClassifierWrapper(Parser):
    """
    Parser that extracts label weights from a label classifier.
    This parser is really meant to be used in conjunction with
    other parsers downstream that make use of these weights.

    If you use it in standalone mode, it will just provide the
    standard unknown prediction everywhere

    Notes
    -----
    *Cache keys*:

    * label: label model path
    """
    def __init__(self, learner):
        """
        Parameters
        ----------
        learner: LabelClassifier
        """
        self._learner = learner

    def fit(self, dpacks, targets, nonfixed_pairs=None, cache=None):
        """
        Extract whatever models or other information from the multipack
        that is necessary to make the labeller operational

        Returns
        -------
        self: object
        """
        # cache management
        cache_file = (cache.get('label') if cache is not None
                      else None)
        # load cached classifier, if it exists
        if cache_file is not None and fp.exists(cache_file):
            # print('\tload {}'.format(cache_file))
            self._learner = joblib.load(cache_file)
            return self

        # filter and modify data: keep only attached
        if nonfixed_pairs is not None:
            attached_pairs = [idxes_attached(dpack, target)
                              for dpack, target in zip(dpacks, targets)]
            # find the indices of the attached pairs that are in
            # nonfixed_pairs
            att_nf_pairs = [np.in1d(att_pairs, nf_pairs)
                            for att_pairs, nf_pairs
                            in zip(attached_pairs, nonfixed_pairs)]
            nonfixed_pairs = [np.where(anf_pairs)[0]
                              for anf_pairs in att_nf_pairs]

        dpacks, targets = self.dzip(attached_only, dpacks, targets)
        dpacks, targets = self.dzip(for_labelling, dpacks, targets)
        self._learner.fit(dpacks, targets, nonfixed_pairs=nonfixed_pairs)
        # save classifier, if necessary
        if cache_file is not None:
            # print('\tsave {}'.format(cache_file))
            joblib.dump(self._learner, cache_file)
        return self

    def transform(self, dpack, nonfixed_pairs=None):
        dpack, _ = for_labelling(dpack, dpack.target)
        weights_l = self._learner.predict_score(
            dpack, nonfixed_pairs=nonfixed_pairs)
        dpack = self.multiply(dpack, label=weights_l)
        return dpack


class SimpleLabeller(LabelClassifierWrapper):
    """
    A simple parser that assigns the best label to any edges with
    unknown labels.

    This can be used as a standalone parser if the underlying
    classifier predicts UNRELATED.

    Notes
    -----
    *Cache keys*

    label: label model path
    """
    def transform(self, dpack, nonfixed_pairs=None):
        dpack = super(SimpleLabeller, self).transform(
            dpack, nonfixed_pairs=nonfixed_pairs)
        new_best_lbls = np.argmax(dpack.graph.label, axis=1)
        unk_lbl = dpack.label_number(UNKNOWN)
        prediction_ = (new if old == unk_lbl else old
                       for old, new in
                       zip(dpack.graph.prediction, new_best_lbls))
        prediction = np.fromiter(prediction_, dtype=np.dtype(np.int16))
        graph = dpack.graph.tweak(prediction=prediction)
        dpack = dpack.set_graph(graph)
        return dpack
