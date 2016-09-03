"""
A parser that only decides on the attachment task (whether this
is directed or not depends on the underlying datapack and decoder).
You could also combine this with the label parser
"""

from __future__ import print_function

from os import path as fp
import sys

import joblib

from attelo.table import for_attachment
from .interface import Parser
from .pipeline import Pipeline

# pylint: disable=too-few-public-methods


class AttachClassifierWrapper(Parser):
    """
    Parser that extracts attachments weights from an attachment
    classifier.

    This parser is really meant to be used in conjunction with
    other parsers downstream that make use of these weights.

    If you use it in standalone mode, it will just provide the
    standard unknown prediction everywhere

    Notes
    -----
    *Cache keys*

    * attach: attachment model path
    """
    def __init__(self, learner_attach):
        """
        Parameters
        ----------
        attach_learner : AttachClassifier
        """
        self._learner_attach = learner_attach

    def fit(self, dpacks, targets, nonfixed_pairs=None, cache=None):
        """
        Extract whatever models or other information from the multipack
        that is necessary to make the parser operational

        Parameters
        ----------
        mpack : MultiPack
        """
        cache_file = (cache.get('attach') if cache is not None
                      else None)
        # load cached classifier, if it exists
        if cache_file is not None and fp.exists(cache_file):
            # print('\tload {}'.format(cache_file))
            self._learner_attach = joblib.load(cache_file)
            return self

        dpacks, targets = self.dzip(for_attachment, dpacks, targets)

        # WIP select only the nonfixed pairs
        if nonfixed_pairs is not None:
            dpacks = [dpack.selected(nf_pairs)
                      for dpack, nf_pairs in zip(dpacks, nonfixed_pairs)]
            targets = [target[nf_pairs]
                       for target, nf_pairs in zip(targets, nonfixed_pairs)]

        self._learner_attach.fit(dpacks, targets)
        # save classifier, if necessary
        if cache_file is not None:
            print('dump attach model to {}'.format(cache_file))  # DEBUG
            sys.stdout.flush()  # DEBUG
            joblib.dump(self._learner_attach, cache_file)
            print('... done')  # DEBUG
            sys.stdout.flush()  # DEBUG
        return self

    def transform(self, dpack, nonfixed_pairs=None):
        attach_pack, _ = for_attachment(dpack, dpack.target)
        # WIP pass only nonfixed pairs to the classifier
        if nonfixed_pairs is not None:
            attach_pack = attach_pack.selected(nonfixed_pairs)
        # end nonfixed_pairs
        weights_a = self._learner_attach.predict_score(attach_pack)
        # WIP overwrite only the attachment scores of non-fixed pairs
        if nonfixed_pairs is not None:
            scores = np.copy(dpack.graph.attach)
            scores[nonfixed_pairs] = weights_a
        else:
            scores = weights_a
        dpack = self.multiply(dpack, attach=scores)
        return dpack


class AttachPipeline(Pipeline):
    """
    Parser that perform the attachment task (may be directed
    or undirected depending on datapack and models)

    For the moment, this assumes AD models, but perhaps over
    time could be generalised to A.D models too

    This can work as a standalone parser: if the datapack is
    unweighted it will initalise it from the classifier.
    Also, if there are pre-existing weights, they will be
    multiplied with the new weights

    Notes
    -----
    *Cache keys*:

    * attach: attachment model path
    """
    def __init__(self, learner, decoder):
        """
        Parameters
        ----------
        learner: AttachClassifier
        labeller: Labeller
        decoder: Decoder
        """
        steps = [('attach weights', AttachClassifierWrapper(learner)),
                 ('decoder', decoder)]
        super(AttachPipeline, self).__init__(steps=steps)
