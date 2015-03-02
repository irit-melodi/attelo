'''
graphing output from the harness
'''

from __future__ import print_function
from enum import Enum
from os import path as fp
import argparse
import sys

from attelo.harness.config import CliArgs
from attelo.io import Torpor
from joblib import (Parallel, delayed)
import attelo.cmd.graph

from .local import (GRAPH_DOCS,
                    DETAILED_EVALUATIONS)
from .path import (decode_output_path,
                   edu_input_path,
                   features_path,
                   fold_dir_basename,
                   pairings_path,
                   report_dir_path)

# pylint: disable=too-few-public-methods


class GoldGraphArgs(CliArgs):
    'cmd line args to generate graphs (gold set)'
    def __init__(self, lconf):
        self.lconf = lconf
        super(GoldGraphArgs, self).__init__()

    def parser(self):
        psr = argparse.ArgumentParser()
        attelo.cmd.graph.config_argparser(psr)
        return psr

    def argv(self):
        lconf = self.lconf
        has_stripped = fp.exists(features_path(lconf, stripped=True))
        args = [edu_input_path(lconf),
                '--quiet',
                '--gold',
                pairings_path(lconf),
                features_path(lconf, stripped=has_stripped),
                '--output',
                fp.join(report_dir_path(lconf, None),
                        'graphs-gold')]
        if GRAPH_DOCS is not None:
            args.extend(['--select'])
            args.extend(GRAPH_DOCS)
        return args


class GraphDiffMode(Enum):
    "what sort of graph output to make"
    solo = 1
    diff = 2
    diff_intra = 3


class GraphArgs(CliArgs):
    'cmd line args to generate graphs (for a fold)'
    def __init__(self, lconf, econf, fold, diffmode):
        self.lconf = lconf
        self.econf = econf
        self.fold = fold
        self.diffmode = diffmode
        super(GraphArgs, self).__init__()

    def parser(self):
        psr = argparse.ArgumentParser()
        attelo.cmd.graph.config_argparser(psr)
        return psr

    def argv(self):
        lconf = self.lconf
        econf = self.econf
        fold = self.fold

        args = [edu_input_path(lconf),
                '--graphviz-timeout', str(15),
                '--quiet']

        if self.diffmode == GraphDiffMode.solo:
            output_bn_prefix = 'graphs-'
            args.extend(['--predictions',
                         decode_output_path(lconf, econf, fold)])
        else:
            has_stripped = fp.exists(features_path(lconf, stripped=True))
            output_bn_prefix = 'graphs-gold-vs-'
            args.extend(['--gold',
                         pairings_path(lconf),
                         features_path(lconf, stripped=has_stripped),
                         '--diff-to',
                         decode_output_path(lconf, econf, fold)])

        if self.diffmode == GraphDiffMode.diff_intra:
            output_bn_prefix = 'graphs-sent-gold-vs-'
            args.extend(['--intra'])

        output_path = fp.join(report_dir_path(lconf, None),
                              output_bn_prefix + fold_dir_basename(fold),
                              econf.key)
        args.extend(['--output', output_path])
        if GRAPH_DOCS is not None:
            args.extend(['--select'])
            args.extend(GRAPH_DOCS)
        return args


def _mk_econf_graphs(lconf, econf, fold, diff):
    "Generate graphs for a single configuration"
    with GraphArgs(lconf, econf, fold, diff) as args:
        attelo.cmd.graph.main_for_harness(args)


def mk_graphs(lconf, dconf):
    "Generate graphs for the gold data and for one of the folds"
    with GoldGraphArgs(lconf) as args:
        if fp.exists(args.output):
            print("skipping gold graphs (already done)",
                  file=sys.stderr)
        else:
            with Torpor('creating gold graphs'):
                attelo.cmd.graph.main_for_harness(args)
    fold = sorted(set(dconf.folds.values()))[0]

    with Torpor('creating graphs for fold {}'.format(fold),
                sameline=False):
        jobs = []
        for mode in GraphDiffMode:
            jobs.extend([delayed(_mk_econf_graphs)(lconf, econf, fold, mode)
                         for econf in DETAILED_EVALUATIONS])
        Parallel(n_jobs=-1)(jobs)
