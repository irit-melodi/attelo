"""Metrics to assess performance on tree-structured predictions.

Functions named as ``*_loss`` return a scalar value to minimize:
the lower the better.
"""

# TODO check that ref_tree and pred_tree both define (strictly?) one
# incoming edge per node


def tree_loss(ref_tree, pred_tree):
    """Compute the tree loss.

    The tree loss is the fraction of edges that are incorrectly predicted.

    Parameters
    ----------
    ref_tree: list of edges (source, target, label)
        reference tree
    pred_tree: list of edges (source, target, label)
        predicted tree

    Returns
    -------
    loss: float
        Return the tree loss between edges of ``ref_tree`` and
        ``pred_tree``.

    See also
    --------
    labelled_tree_loss

    Notes
    -----
    For labelled trees, the tree loss checks for strict correspondence:
    it does not differentiate between incorrectly attached edges and
    correctly attached but incorrectly labelled edges.
    """
    return 1.0 - (len(set(pred_tree) & set(ref_tree))/ float(len(ref_tree)))


def labelled_tree_loss(ref_tree, pred_tree):
    """Compute the labelled tree loss.

    The labelled tree loss is the fraction of edges that are incorrectly
    predicted, with a lesser penalty for edges with the correct attachment
    but the wrong label.

    Parameters
    ----------
    ref_tree: list of edges (source, target, label)
        reference tree
    pred_tree: list of edges (source, target, label)
        predicted tree

    Returns
    -------
    loss: float
        Return the tree loss between edges of ``ref_tree`` and
        ``pred_tree``.

    See also
    --------
    tree_loss

    Notes
    -----
    The labelled tree loss counts only half of the penalty for edges with
    the right attachment but the wrong label.
    """
    edges_ref = {tgt: (src, lbl) for src, tgt, lbl in ref_tree}
    edges_pred = {tgt: (src, lbl) for src, tgt, lbl in pred_tree}
    
    score = 0.0
    for tgt in sorted(set(edges_ref) & set(edges_pred)):
        head_ref, lbl_ref = edges_ref[tgt]
        head_pred, lbl_pred = edges_pred[tgt]
        if head_pred == head_ref:
            score += 0.5
            if lbl_pred == lbl_ref:
                score += 0.5

    return 1.0 - score / len(ref_tree)


# TODO hinge loss for tree-structured prediction ; this implies refactoring
# attelo.learning.perceptron

# TODO refactor and link here some of the functions in attelo.score
