"""
A parser that only decides on the attachment task (whether this
is directed or not depends on the underlying datapack and decoder).
You could also combine this with the label parser
"""

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

    Caveats
    -------
    For the moment, parsers work with cached models only. There is a fit
    function which could be used to pull things out of the training datapacks,
    but it does not currently act on the learners. This could be activated once
    we work out what to do about sharing models across parsers (for example, to
    forget about it altogether)
    """
    def __init__(self, learner_attach):
        """
        Parameters
        ----------
        attach_learner: AttachClassifier
        """
        self._learner_attach = learner_attach

    def fit(self, dpacks, targets):
        """
        Extract whatever models or other information from the multipack
        that is necessary to make the parser operational

        Parameters
        ----------
        mpack : MultiPack
        """
        dpacks, targets = self.dzip(for_attachment, dpacks, targets)
        self._learner_attach.fit(dpacks, targets)

    def transform(self, dpack):
        attach_pack, _ = for_attachment(dpack, dpack.target)
        weights_a = self._learner_attach.transform(attach_pack)
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

    Caveats
    -------
    For the moment, parsers work with cached models only. There is a fit
    function which could be used to pull things out of the training datapacks,
    but it does not currently act on the learners. This could be activated once
    we work out what to do about sharing models across parsers (for example, to
    forget about it altogether)
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
