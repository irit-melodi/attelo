"""Tests for metrics
"""

from .deptree import (compute_uas_las, compute_uas_las_listcomp,
                      compute_uas_las_np)


class FakeRstDepTree(object):
    """Mockup object for `educe.rst_dt.deptree.RstDepTree`.

    This is to add proper tests without attelo depending on educe.
    """
    def __init__(self, heads, labels):
        self.heads = heads
        self.labels = labels


DT_TRUE = FakeRstDepTree([-1, 0, 1, 2, 1],
                         [None, 'xxx', 'yyy', 'zzz', 'ttt'])
# one error on heads
DT_PRED_HD_ERR = FakeRstDepTree([-1, 0, 1, 2, 2],
                                [None, 'xxx', 'yyy', 'zzz', 'ttt'])
# one error on label
DT_PRED_LB_ERR = FakeRstDepTree([-1, 0, 1, 2, 1],
                                [None, 'xxx', 'yyy', 'xxx', 'ttt'])
# one error on both head and label
DT_PRED_BO_ERR = FakeRstDepTree([-1, 0, 1, 2, 2],
                                [None, 'xxx', 'yyy', 'xxx', 'ttt'])


def test_compute_uas_las():
    """Basic tests for compute_uas_las"""
    dtree_true = [DT_TRUE]
    # compare gold with itself: perfect scores
    assert compute_uas_las(dtree_true, dtree_true) == (1.0, 1.0, 1.0)
    # one error on head
    dtree_pred = [DT_PRED_HD_ERR]
    assert compute_uas_las(dtree_true, dtree_pred) == (0.75, 0.75, 1.0)
    # one error on label
    dtree_pred = [DT_PRED_LB_ERR]
    assert compute_uas_las(dtree_true, dtree_pred) == (1.0, 0.75, 0.75)
    # one error on both
    dtree_pred = [DT_PRED_BO_ERR]
    assert compute_uas_las(dtree_true, dtree_pred) == (0.75, 0.5, 0.75)


def test_compute_uas_las_listcomp():
    """Basic tests for compute_uas_las_listcomp"""
    dtree_true = [DT_TRUE]
    # compare gold with itself: perfect scores
    assert (compute_uas_las_listcomp(dtree_true, dtree_true) ==
            (1.0, 1.0, 1.0))
    # one error on head
    dtree_pred = [DT_PRED_HD_ERR]
    assert (compute_uas_las_listcomp(dtree_true, dtree_pred) ==
            (0.75, 0.75, 1.0))
    # one error on label
    dtree_pred = [DT_PRED_LB_ERR]
    assert (compute_uas_las_listcomp(dtree_true, dtree_pred) ==
            (1.0, 0.75, 0.75))
    # one error on both
    dtree_pred = [DT_PRED_BO_ERR]
    assert (compute_uas_las_listcomp(dtree_true, dtree_pred) ==
            (0.75, 0.5, 0.75))


def test_compute_uas_las_np():
    """Basic tests for compute_uas_las_np"""
    dtree_true = [DT_TRUE]
    # compare gold with itself: perfect scores
    assert compute_uas_las_np(dtree_true, dtree_true) == (1.0, 1.0, 1.0)
    # one error on head
    dtree_pred = [DT_PRED_HD_ERR]
    assert compute_uas_las_np(dtree_true, dtree_pred) == (0.75, 0.75, 1.0)
    # one error on label
    dtree_pred = [DT_PRED_LB_ERR]
    assert compute_uas_las_np(dtree_true, dtree_pred) == (1.0, 0.75, 0.75)
    # one error on both
    dtree_pred = [DT_PRED_BO_ERR]
    assert compute_uas_las_np(dtree_true, dtree_pred) == (0.75, 0.5, 0.75)
