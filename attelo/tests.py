"""
attelo tests
"""

# pylint: disable=too-few-public-methods, no-self-use, no-member
# no-member: numpy

from __future__ import print_function
import unittest

import scipy.sparse
import numpy
import numpy as np

import attelo
import attelo.fold

from .edu import EDU, FAKE_ROOT
from .fold import select_training
from .table import (DataPack,
                    DataPackException,
                    attached_only,
                    groupings)

MAX_FOLDS = 2


def squish(matrix):
    'convert a sparse matrix to list'
    return matrix.todense().flatten().tolist()


class DataPackTest(unittest.TestCase):
    '''
    basic tests on data pack filtering operations
    '''
    edus = [EDU('e1', 'hi', 0, 1, 'a', 's1'),
            EDU('e2', 'there', 3, 8, 'a', 's1'),
            EDU('e3', 'you', 9, 12, 'a', 's2')]
    trivial = DataPack(edus=edus,
                       pairings=[(edus[0], edus[1])],
                       data=scipy.sparse.csr_matrix([[6, 8]]),
                       target=numpy.array([1]),
                       ctarget=dict(),  # DIRTY
                       labels=['__UNK__', 'x', 'UNRELATED'],
                       graph=None,
                       vocab=None)
    trivial_bidi = DataPack(edus,
                            pairings=[(edus[0], edus[1]),
                                      (edus[1], edus[0])],
                            data=scipy.sparse.csr_matrix([[6, 8],
                                                          [7, 0]]),
                            target=numpy.array([1, 0]),
                            ctarget=dict(),  # DIRTY
                            labels=['__UNK__', 'x', 'UNRELATED'],
                            graph=None,
                            vocab=None)

    # pylint: disable=invalid-name
    def assertEqualishDatapack(self, pack1, pack2):
        '''
        series of assertions that two datapacks are
        equivalant enough for our tests
        '''
        self.assertEqual(pack1.edus, pack2.edus)
        self.assertEqual(pack1.pairings, pack2.pairings)
        self.assertEqual(pack1.labels, pack2.labels)
        self.assertEqual(pack1.target.tolist(), pack2.target.tolist())
        self.assertEqual(pack1.data.shape, pack2.data.shape)
        self.assertEqual(squish(pack1.data), squish(pack2.data))

    def assertEqualEduIds(self, pack, ids):
        '''
        the datapack or multipack has all the of the edus with the
        '''
        if isinstance(pack, DataPack):
            edus = frozenset(e.id for e in pack.edus)
        else:
            edus = frozenset(e.id
                             for p in pack.values()
                             for e in p.edus)
        self.assertEqual(edus, frozenset(ids))
    # pylint: enable=invalid-name

    def test_trivial_sanity(self):
        'can build a full data pack from the trivial one'
        triv = self.trivial
        self.assertRaises(DataPackException, DataPack.load,
                          triv.edus,
                          triv.pairings,
                          triv.data,
                          [1, 1],
                          dict(),  # ctarget, DIRTY
                          ['__UNK__', 'UNRELATED', 'foo'],
                          None)

        # check grouping of edus
        fake1 = EDU(self.edus[1].id,
                    self.edus[1].text,
                    self.edus[1].start,
                    self.edus[1].end,
                    'b',
                    's2')
        self.assertRaises(DataPackException, groupings,
                          [(self.edus[0], fake1)])
        # but root is ok
        self.assertTrue(DataPack.load([self.edus[0]],
                                      [(self.edus[0], FAKE_ROOT)],
                                      triv.data,
                                      triv.target,
                                      dict(),  # ctarget: DIRTY
                                      triv.labels,
                                      None))
        dpack2 = DataPack.load(triv.edus,
                               triv.pairings,
                               triv.data,
                               triv.target,
                               dict(),  # ctarget: DIRTY
                               triv.labels,
                               None)
        self.assertEqualishDatapack(triv, dpack2)

    def test_get_label(self):
        'correctly picks out labels and unrelated'
        pack = DataPack(self.edus,
                        pairings=[(self.edus[0], self.edus[1]),
                                  (self.edus[1], self.edus[0]),
                                  (self.edus[0], self.edus[2]),
                                  (self.edus[2], self.edus[0])],
                        data=scipy.sparse.csr_matrix([[6], [7], [1], [5]]),
                        target=numpy.array([2, 1, 1, 3]),
                        ctarget=dict(),  # DIRTY
                        labels=['__UNK__', 'x', 'y', 'UNRELATED'],
                        graph=None,
                        vocab=None)
        labels = [pack.get_label(t) for t in pack.target]
        self.assertEqual(['y', 'x', 'x', 'UNRELATED'], labels)


    def test_select_classes(self):
        'test that classes are filtered correctly'
        # pylint: disable=invalid-name
        a1 = EDU('a1', 'hi', 0, 1, 'a', 's1')
        a2 = EDU('a2', 'there', 3, 8, 'a', 's1')
        b1 = EDU('b1', 'this', 0, 4, 'b', 's2')
        b2 = EDU('b2', 'is', 6, 8, 'b', 's2')
        # pylint: enable=invalid-name

        orig_classes = ['__UNK__', 'there', 'are', 'four', 'UNRELATED', 'lights']
        pack = DataPack.load(edus=[a1, a2,
                                   b1, b2],
                             pairings=[(a1, a2),
                                       (b1, b2),
                                       (b1, FAKE_ROOT)],
                             data=scipy.sparse.csr_matrix([[6, 8],
                                                           [7, 0],
                                                           [3, 9]]),
                             target=numpy.array([3, 4, 2]),
                             ctarget=dict(),  # DIRTY
                             labels=orig_classes,
                             vocab=None)

        pack1, _ = attached_only(pack, pack.target)
        self.assertEqual(orig_classes, pack1.labels)
        self.assertEqual(list(pack1.target), [3, 2])

        pack2 = pack.selected([0, 1])
        self.assertEqual(orig_classes, pack2.labels)

        pack3 = pack.selected([1, 2])
        self.assertEqual(orig_classes, pack3.labels)

    def test_folds(self):
        'test that fold selection does something sensible'

        # pylint: disable=invalid-name
        a1 = EDU('a1', 'hi', 0, 1, 'a', 's1')
        a2 = EDU('a2', 'there', 3, 8, 'a', 's1')
        b1 = EDU('b1', 'this', 0, 4, 'b', 's2')
        b2 = EDU('b2', 'is', 6, 8, 'b', 's2')
        c1 = EDU('c1', 'rather', 0, 7, 'c', 's3')
        c2 = EDU('c2', 'tedious', 9, 16, 'c', 's3')
        d1 = EDU('d1', 'innit', 0, 5, 'd', 's4')
        d2 = EDU('d2', '?', 6, 7, 'd', 's4')
        # pylint: enable=invalid-name

        labels = ['__UNK__', 'x', 'y', 'UNRELATED']
        mpack = {'a': DataPack.load(edus=[a1, a2],
                                    pairings=[(a1, a2)],
                                    data=scipy.sparse.csr_matrix([[6, 8]]),
                                    target=numpy.array([1]),
                                    ctarget=dict(),  # DIRTY
                                    labels=labels,
                                    vocab=None),
                 'b': DataPack.load(edus=[b1, b2],
                                    pairings=[(b1, b2),
                                              (b1, FAKE_ROOT)],
                                    data=scipy.sparse.csr_matrix([[7, 0],
                                                                  [3, 9]]),
                                    target=numpy.array([0, 1]),
                                    ctarget=dict(),  # DIRTY
                                    labels=labels,
                                    vocab=None),
                 'c': DataPack.load(edus=[c1, c2],
                                    pairings=[(c1, c2)],
                                    data=scipy.sparse.csr_matrix([[1, 1]]),
                                    target=numpy.array([1]),
                                    ctarget=dict(),  # DIRTY
                                    labels=labels,
                                    vocab=None),
                 'd': DataPack.load(edus=[d1, d2],
                                    pairings=[(d1, d2)],
                                    data=scipy.sparse.csr_matrix([[0, 4]]),
                                    target=numpy.array([0]),
                                    ctarget=dict(),  # DIRTY
                                    labels=labels,
                                    vocab=None)}
        fold_dict = {'a': 0,
                     'b': 1,
                     'c': 0,
                     'd': 1}
        self.assertEqualEduIds(select_training(mpack, fold_dict, 0),
                               ['b1', 'b2', 'd1', 'd2'])
        self.assertEqualEduIds(select_training(mpack, fold_dict, 1),
                               ['a1', 'a2', 'c1', 'c2'])
        # nose gets confused by the 'test' in select_testing
        # if we import it directly
        self.assertEqualEduIds(attelo.fold.select_testing(mpack, fold_dict, 0),
                               ['a1', 'a2', 'c1', 'c2'])
        self.assertEqualEduIds(attelo.fold.select_testing(mpack, fold_dict, 1),
                               ['b1', 'b2', 'd1', 'd2'])
