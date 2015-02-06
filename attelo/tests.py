"""
attelo tests
"""

# pylint: disable=too-few-public-methods, no-self-use, no-member
# no-member: numpy

from __future__ import print_function
from os import path as fp
import argparse
import csv
import shutil
import tempfile
import unittest

import scipy.sparse
import numpy

from attelo.harness.config import CliArgs
import attelo

from .edu import EDU, FAKE_ROOT
from .table import DataPack, DataPackException


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
                       labels=['x', 'UNRELATED'])
    trivial_bidi = DataPack(edus,
                            pairings=[(edus[0], edus[1]),
                                      (edus[1], edus[0])],
                            data=scipy.sparse.csr_matrix([[6, 8],
                                                          [7, 0]]),
                            target=numpy.array([1, 0]),
                            labels=['x', 'UNRELATED'])

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
        the data pack has all the of the edus with the
        '''
        self.assertEqual(frozenset(e.id for e in pack.edus),
                         frozenset(ids))
    # pylint: enable=invalid-name

    def test_trivial_sanity(self):
        'can build a full data pack from the trivial one'
        triv = self.trivial
        self.assertRaises(DataPackException, DataPack.load,
                          triv.edus,
                          triv.pairings,
                          triv.data,
                          [1, 1],
                          ['UNRELATED', 'foo'])

        # check grouping of edus
        fake1 = EDU(self.edus[1].id,
                    self.edus[1].text,
                    self.edus[1].start,
                    self.edus[1].end,
                    'b',
                    's2')
        self.assertRaises(DataPackException, DataPack.load,
                          [self.edus[0], fake1],
                          [(self.edus[0], fake1)],
                          triv.data,
                          triv.target,
                          triv.labels)
        # but root is ok
        self.assertTrue(DataPack.load([self.edus[0]],
                                      [(self.edus[0], FAKE_ROOT)],
                                      triv.data,
                                      triv.target,
                                      triv.labels))
        dpack2 = DataPack.load(triv.edus,
                               triv.pairings,
                               triv.data,
                               triv.target,
                               triv.labels)
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
                        labels=['x', 'y', 'UNRELATED'])
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

        orig_classes = ['there', 'are', 'four', 'UNRELATED', 'lights']
        pack = DataPack.load(edus=[a1, a2,
                                   b1, b2],
                             pairings=[(a1, a2),
                                       (b1, b2),
                                       (b1, FAKE_ROOT)],
                             data=scipy.sparse.csr_matrix([[6, 8],
                                                           [7, 0],
                                                           [3, 9]]),
                             target=numpy.array([3, 4, 2]),
                             labels=orig_classes)

        pack1 = pack.attached_only()
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

        pack = DataPack.load(edus=[a1, a2,
                                   b1, b2,
                                   c1, c2,
                                   d1, d2],
                             pairings=[(a1, a2),
                                       (b1, b2),
                                       (b1, FAKE_ROOT),
                                       (c1, c2),
                                       (d1, d2)],
                             data=scipy.sparse.csr_matrix([[6, 8],
                                                           [7, 0],
                                                           [3, 9],
                                                           [1, 1],
                                                           [0, 4]]),
                             target=numpy.array([1, 0, 1, 1, 0]),
                             labels=['x', 'y', 'UNRELATED'])
        fold_dict = {'a': 0,
                     'b': 1,
                     'c': 0,
                     'd': 1}
        self.assertEqualEduIds(pack.training(fold_dict, 0),
                               ['b1', 'b2', 'd1', 'd2'])
        self.assertEqualEduIds(pack.training(fold_dict, 1),
                               ['a1', 'a2', 'c1', 'c2'])
        self.assertEqualEduIds(pack.testing(fold_dict, 0),
                               ['a1', 'a2', 'c1', 'c2'])
        self.assertEqualEduIds(pack.testing(fold_dict, 1),
                               ['b1', 'b2', 'd1', 'd2'])


# ---------------------------------------------------------------------
#
# ---------------------------------------------------------------------

class TmpDir(object):
    "crude context manager for creating, deleting tmpdir"
    def __init__(self):
        self.tmpdir = tempfile.mkdtemp()

    def __enter__(self):
        return self.tmpdir

    # pylint: disable=unused-argument
    def __exit__(self, ctype, value, traceback):
        shutil.rmtree(self.tmpdir)
    # pylint: enable=unused-argument


class TestArgs(CliArgs):
    "arguments in test harness"

    def __init__(self, tmpdir):
        self._tmpdir = tmpdir
        super(TestArgs, self).__init__()

    def parser(self):
        psr = argparse.ArgumentParser()
        self.module().config_argparser(psr)
        return psr

    @classmethod
    def module(cls):
        'attelo command module'
        raise NotImplementedError()

    def eg_path(self, subpath):
        "path to example dir"
        return fp.join('example', subpath)

    def tmp_path(self, subpath):
        "path within tmpdir"
        return fp.join(self._tmpdir, subpath)

    def argv(self):
        "command line args"
        return [self.eg_path('tiny.edus'),
                self.eg_path('tiny.pairings'),
                self.eg_path('tiny.features.sparse')]

    @classmethod
    def run(cls, *args, **kwargs):
        "run the attelo command that goes with these args"
        with cls(*args, **kwargs) as cli_args:
            cls.module().main(cli_args)


class EvaluateArgs(TestArgs):
    "args to attelo evaluate"

    def __init__(self, *args, **kwargs):
        super(EvaluateArgs, self).__init__(*args, **kwargs)

    def argv(self):
        args = super(EvaluateArgs, self).argv()
        args.extend(['--config', self.eg_path('tiny.config'),
                     '--nfold', str(MAX_FOLDS)])
        return args

    @classmethod
    def module(cls):
        return attelo.cmd.evaluate


class EnfoldArgs(TestArgs):
    "args to attelo enfold"

    # pylint: disable=unused-argument
    def __init__(self, *args, **kwargs):
        super(EnfoldArgs, self).__init__(*args, **kwargs)
    # pylint: enable=unused-argument

    def argv(self):
        args = super(EnfoldArgs, self).argv()
        args += ['--config', self.eg_path('tiny.config'),
                 '--nfold', str(MAX_FOLDS),
                 '--output', self.tmp_path('folds.json')]
        return args

    @classmethod
    def module(cls):
        return attelo.cmd.enfold


class LearnDecodeArgs(TestArgs):
    "args to either attelo learn or decode"
    # pylint: disable=unused-argument
    def __init__(self, fold=None, extra_args=None, *args, **kwargs):
        self._fold = fold
        self._extra_args = extra_args or []
        super(LearnDecodeArgs, self).__init__(*args, **kwargs)
    # pylint: enable=unused-argument

    def argv(self):
        args = super(LearnDecodeArgs, self).argv()
        if self._fold is not None:
            args.extend(['--fold-file', self.tmp_path('folds.json'),
                         '--fold', str(self._fold)])
        args.extend(['--config', self.eg_path('tiny.config'),
                     '--quiet',
                     '--attachment-model', self.tmp_path('attach.model'),
                     '--relation-model', self.tmp_path('relate.model')])
        args.extend(self._extra_args)
        return args

    @classmethod
    def module(cls):
        return NotImplementedError


class LearnArgs(LearnDecodeArgs):
    "args to attelo learn"

    # pylint: disable=unused-argument
    def __init__(self, *args, **kwargs):
        super(LearnArgs, self).__init__(*args, **kwargs)
    # pylint: enable=unused-argument

    @classmethod
    def module(cls):
        return attelo.cmd.learn


class DecodeArgs(LearnDecodeArgs):
    "args to attelo decode"

    # pylint: disable=unused-argument
    def __init__(self, *args, **kwargs):
        super(DecodeArgs, self).__init__(*args, **kwargs)
    # pylint: enable=unused-argument

    def counts_file(self):
        'file where scores will be stored'
        bname = 'scores'
        if self._fold is not None:
            bname += '-{}'.format(self._fold)
        return self.tmp_path(bname)

    def argv(self):
        args = super(DecodeArgs, self).argv()
        args.extend(['--output', fp.join(self._tmpdir, 'output'),
                     '--scores', self.counts_file()])
        return args

    @classmethod
    def module(cls):
        return attelo.cmd.decode


class ReportArgs(TestArgs):
    "args to attelo report"

    def __init__(self, idx_path, *args, **kwargs):
        self._idx_path = idx_path
        super(ReportArgs, self).__init__(*args, **kwargs)

    def argv(self):
        return [self._idx_path]

    @classmethod
    def module(cls):
        return attelo.cmd.report


def fake_harness(*args, **kwargs):
    '''sequence of attelo commands that fit together like they
    might in a harness'''
    EnfoldArgs.run(*args, **kwargs)

    for i in range(0, MAX_FOLDS):
        LearnArgs.run(*args, fold=i, **kwargs)
        DecodeArgs.run(*args, fold=i, **kwargs)
    #ReportArgs.run(idx_path=idx_filename, *args, **kwargs)


class CliTest(unittest.TestCase):
    """
    Run command line utilities on sample data
    """
    def _vary(self, test, **kwargs):
        '''
        run a test through whatever systematic variations we can
        think of
        '''
        with TmpDir() as tmpdir:
            test(tmpdir=tmpdir, **kwargs)

    def test_evaluate(self):
        'attelo evaluate'
        self._vary(EvaluateArgs.run)

    def test_enfold(self):
        'attelo enfold'
        self._vary(EnfoldArgs.run)

    def test_learn(self):
        'attelo learn'
        self._vary(LearnArgs.run)
        for learner in ["maxent",
                        "svm",
                        "majority",
                        "bayes"]:
            self._vary(LearnArgs.run, extra_args=["--learner", learner])

    def test_harness(self):
        'attelo enfold, learn, decode, report'
        self._vary(fake_harness)
