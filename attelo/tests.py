"""
attelo tests
"""

# pylint: disable=too-few-public-methods, no-self-use

from __future__ import print_function
from os import path as fp
import argparse
import csv
import shutil
import tempfile
import unittest

from attelo.harness.config import CliArgs
import attelo


MAX_FOLDS = 2


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

    def __init__(self, tmpdir, relate):
        self._tmpdir = tmpdir
        self._relate = relate
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
        args = [self.eg_path('tiny.attach.tab')]
        if self._relate:
            args.extend([self.eg_path('tiny.relate.tab')])
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
        return [self.eg_path('tiny.attach.tab'),
                '--config', self.eg_path('tiny.config'),
                '--nfold', str(MAX_FOLDS),
                '--output', self.tmp_path('folds.json')]

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
        args = [self.eg_path('tiny.attach.tab')]
        if self._relate:
            args.extend([self.eg_path('tiny.relate.tab'),
                         '--relation-model', self.tmp_path('relate.model')])
        if self._fold is not None:
            args.extend(['--fold-file', self.tmp_path('folds.json'),
                         '--fold', str(self._fold)])
        args.extend(['--config', self.eg_path('tiny.config'),
                     '--quiet',
                     '--attachment-model', self.tmp_path('attach.model')])
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
        args.extend(['--output', self._tmpdir,
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
    tmpdir = kwargs['tmpdir']
    idx_filename = fp.join(tmpdir, 'index.csv')

    with open(idx_filename, 'w') as idxraw:
        idx = csv.writer(idxraw)
        idx.writerow(['config', 'fold', 'counts_file'])
        for i in range(0, MAX_FOLDS):
            LearnArgs.run(*args, fold=i, **kwargs)
            DecodeArgs.run(*args, fold=i, **kwargs)
            scores_file = fp.join(tmpdir, 'scores-{}'.format(i))
            idx.writerow(['nose', str(i), scores_file])
    ReportArgs.run(idx_path=idx_filename, *args, **kwargs)


class CliTest(unittest.TestCase):
    """
    Run command line utilities on sample data
    """
    def _vary(self, test, **kwargs):
        'run a test both with and without rels'
        with TmpDir() as tmpdir:
            test(relate=False, tmpdir=tmpdir, **kwargs)
        with TmpDir() as tmpdir:
            test(relate=True, tmpdir=tmpdir, **kwargs)

    def test_evaluate(self):
        'attelo evaluate'
        self._vary(EvaluateArgs.run)

    def test_enfold(self):
        'attelo enfold'
        self._vary(EnfoldArgs.run)

    def test_learn(self):
        'attelo learn'
        self._vary(LearnArgs.run)

    def test_learn_maxent(self):
        'attelo learn'
        self._vary(LearnArgs.run, extra_args=["--learner", "maxent"])

    def test_harness(self):
        'attelo enfold, learn, decode, report'
        self._vary(fake_harness)
