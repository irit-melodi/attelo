"""
A parser that only decides on the attachment task (whether this
is directed or not depends on the underlying datapack and decoder).
You could also combine this with the label parser
"""

from os import path as fp

from attelo.io import (load_model, save_model)
from attelo.table import (for_attachment)
from .interface import (Parser)
from .pipeline import (Pipeline)

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
        attach_learner: AttachClassifier
        """
        self._learner_attach = learner_attach

    def fit(self, dpacks, targets, cache=None):
        """
        Extract whatever models or other information from the multipack
        that is necessary to make the parser operational

        Parameters
        ----------
        mpack : MultiPack
        """
        cache = cache or {}
        cache_file = cache.get('attach')
        if cache_file is not None and fp.exists(cache_file):
            self._learner_attach = load_model(cache_file)
            return self
        else:
            dpacks, targets = self.dzip(for_attachment, dpacks, targets)
            self._learner_attach.fit(dpacks, targets)
            if cache_file is not None:
                save_model(cache_file, self._learner_attach)
            return self

    def transform(self, dpack):
        attach_pack, _ = for_attachment(dpack, dpack.target)
        weights_a = self._learner_attach.predict_score(attach_pack)
        return self.multiply(dpack, attach=weights_a)


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
