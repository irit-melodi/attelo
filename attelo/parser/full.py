"""
A 'full' parser does the attach, direction, and labelling tasks
"""

import numpy as np

from .attach import AttachClassifierWrapper
from .label import (LabelClassifierWrapper, SimpleLabeller)
from .interface import (Parser)
from .pipeline import (Pipeline)

# pylint: disable=too-few-public-methods

class AttachTimesBestLabel(Parser):
    """
    Intermediary parser that adjusts the attachment weight
    by multiplying the best label weight with it.

    This is most useful in the middle of a parsing pipeline:
    we need something upstream to assign initial attachment and
    label weights (otherwise we get the default 1.0 everywhere),
    and something downstream to make predictions (otherwise
    it's UNKNOWN everywhere)
    """
    def fit(self, dpacks, targets):
        return

    def transform(self, dpack):
        dpack = self.multiply(dpack)
        weights_a = dpack.graph.attach
        weights_l = dpack.graph.label
        weights_best_label = np.ravel(np.amax(weights_l, axis=1))
        weights_a = np.multiply(weights_a, weights_best_label)
        graph = dpack.graph.tweak(attach=weights_a)
        return dpack.set_graph(graph)


class JointPipeline(Pipeline):
    """
    Parser that performs attach, direction, and labelling tasks.

    For the moment, this assumes AD.L models, but we hope to
    explore possible generalisations of this idea over time.

    In our working shorthand, this would be an AD.L:adl parser,
    ie. one that has separate attach-direct model and label
    model (AD.L); but which treats decoding as a joint-prediction
    task.

    Caveats
    -------
    For the moment, parsers work with cached models only. There is a fit
    function which could be used to pull things out of the training datapacks,
    but it does not currently act on the learners. This could be activated once
    we work out what to do about sharing models across parsers (for example, to
    forget about it altogether)
    """
    def __init__(self,
                 learner_attach,
                 learner_label,
                 decoder):
        """
        Parameters
        ----------
        attach_learner: AttachClassifier
        label_learner: LabelClassifier
        decoder: Decoder
        """
        if not learner_attach.can_predict_proba:
            raise ValueError('Attachment model does not know how to predict '
                             'probabilities.')
        if not learner_label.can_predict_proba:
            raise ValueError('Relation labelling model does not '
                             'know how to predict probabilities')
        steps = [('attach weights', AttachClassifierWrapper(learner_attach)),
                 ('label weights', LabelClassifierWrapper(learner_label)),
                 ('attach x best label', AttachTimesBestLabel()),
                 ('decoder', decoder)]
        super(JointPipeline, self).__init__(steps=steps)


class PostlabelPipeline(Pipeline):
    """
    Parser that perform the attachment task (may be directed
    or undirected depending on datapack and models), and then
    the labelling task in a second step

    For the moment, this assumes AD models, but perhaps over
    time could be generalised to A.D models too

    This can work as a standalone parser: if the datapack is
    unweighted it will initalise it from the classifier.
    Also, if there are pre-existing weights, they will be
    multiplied with the new weights
    """
    def __init__(self,
                 learner_attach,
                 learner_label,
                 decoder):
        """
        Parameters
        ----------
        attach_learner: AttachClassifier
        label_learner: LabelClassifier
        decoder: Decoder
        """
        steps = [('attach weights', AttachClassifierWrapper(learner_attach)),
                 ('decode', decoder),
                 ('label', SimpleLabeller(learner=learner_label))]
        super(PostlabelPipeline, self).__init__(steps=steps)
