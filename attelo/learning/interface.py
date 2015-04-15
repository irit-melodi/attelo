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
