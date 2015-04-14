"""
Common interface(s) to Attelo classifiers.
"""

from abc import ABCMeta, abstractmethod
from six import with_metaclass


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
    def fit(self, mpack):
        """
        Learns a classifier and the labels attribute from a
        multipack of documents

        Parameters
        ----------
        mpack: Multipack

        Returns
        -------
        self: object
        """
        raise NotImplementedError

    @abstractmethod
    def transform(self, dpack):
        """
        Parameters
        ----------
        dpack: DataPack
            A single document for which we would like to predict
            labels

        Returns
        -------
        labels: [string]
            List of labels used in the scores array

        scores: array(float)
            A 2D array (sample x label) associating each label with
            a score
        """
        return NotImplementedError
