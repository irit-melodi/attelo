"""
Utility classes functions shared by learners
"""

# pylint: disable=no-name-in-module
from numpy import (zeros)
# pylint: enable=no-name-in-module


def relabel(src_labels, src_weights, tgt_labels):
    """Rearrange the columns of a weight matrix to correspond to
    the new target label layout.

    Target labels must be a superset of the source labels.

    Parameters
    ----------
    src_labels : iterable of int
        List of source labels

    src_weights : 2D matrix of float
        Scores for each pairing and each possible label

    tgt_labels : iterable of int
        List of target labels

    Returns
    -------
    tgt_weights : 2D matrix of float
        Projection of src_weights with reordered columns
    """
    missing = [x for x in src_labels if x not in tgt_labels]
    if missing:
        oops = ("Can't relabel from {src} to {tgt} because the labels "
                "{missing} are missing"
                "").format(src=src_labels,
                           tgt=tgt_labels,
                           missing=missing)
        raise ValueError(oops)
    len_samples = src_weights.shape[0]
    tgt_weights = zeros((len_samples, len(tgt_labels)))
    for src_lbl, lbl_str in enumerate(src_labels):
        tgt_lbl = tgt_labels.index(lbl_str)
        tgt_weights[:, tgt_lbl] = src_weights[:, src_lbl]
    return tgt_weights
