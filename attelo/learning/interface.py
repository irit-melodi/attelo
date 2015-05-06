"""
Common interface(s) to Attelo classifiers.
"""

from abc import ABCMeta, abstractmethod
from six import with_metaclass


class AttachClassifier(with_metaclass(ABCMeta, object)):
    '''
    An attachment classfier associates samples with attachment
    probabilities. Attachment classifiers are agnostic to
    whether their samples correspond to directed or undirected
    edges (the only difference as far as the classifiers are
    concerned may be that there are a little over half as many
    edges in the undirected case)

    Attributes
    ----------
    can_predict_proba: bool

        True if scores should be interpreted as probabilities
    '''
    @abstractmethod
    def fit(self, dpacks, targets):
        """
        Learns a classifier from a collection of datapacks

        Parameters
        ----------
        dpacks: [DataPack]

            one datapack per document, each with its own set of
            features (we expect multiple datapacks because this
            allows for a notion of structured learning that
            somehow exploits the grouping of samples)

        targets: [[int]]

            For each datapack, a list of ternary values encoded
            as ints (-1: not attached, 0: unknown, 1: attached).
            Each list must have the same number items as there
            are samples in its its datapack counterpart.


        Returns
        -------
        self: object
        """
        raise NotImplementedError

    @abstractmethod
    def predict_score(self, dpack):
        """
        Parameters
        ----------
        dpack: DataPack
            A single document for which we would like to predict
            labels

        Returns
        -------
        scores: array(float)
            An array (one score per sample)
        """
        return NotImplementedError


class LabelClassifier(with_metaclass(ABCMeta, object)):
    '''
    A label classifier associates labels with scores.

    Decisions are returned as a (sample x labels) array
    with a score

    Attributes
    ----------
    can_predict_proba: bool

        True if scores should be interpreted as probabilities
    '''
    @abstractmethod
    def fit(self, dpacks, targets):
        """
        Learns a classifier and the labels attribute from a
        multipack of documents

        Parameters
        ----------
        dpacks: [DataPack]

            A list of documents

        targets: [[int]]

            For each datapack, a list of label numbers, one per
            sample. All datapacks are expected to use the same
            label numbering scheme. Use `DataPack.get_label`
            to recover the string values.

            Each list must have the same number items as there
            are samples in its its datapack counterpart.


        Returns
        -------
        self: object
        """
        raise NotImplementedError

    @abstractmethod
    def predict_score(self, dpack):
        """
        Parameters
        ----------
        dpack: DataPack
            A single document for which we would like to predict
            labels

        Returns
        -------
        weights: array(float)
            A 2D array (sample x label) associating each label with
            a score. Mind your array dimensions.
        """
        return NotImplementedError
