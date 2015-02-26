"""
Attelo command configuration
"""

from enum import Enum
from os import path as fp
import argparse

from attelo.harness.config import CliArgs
from attelo.util import (Team)
import attelo.cmd as att

from .local import (IntraFlag,
                    ATTELO_CONFIG_FILE,
                    GRAPH_DOCS)
from .path import (decode_output_path,
                   edu_input_path,
                   eval_model_path,
                   features_path,
                   fold_dir_basename,
                   model_info_path,
                   pairings_path,
                   report_dir_path,
                   vocab_path)

# pylint: disable=too-few-public-methods


def _attelo_dpack_args(lconf):
    """
    Return a list of attelo args that correspond to the data pack
    files (edu inputs, pairings, features)
    """
    return [edu_input_path(lconf),
            pairings_path(lconf),
            features_path(lconf)]


def attelo_doc_model_paths(lconf, rconf, fold):
    """
    Return attelo intra/intersentential model paths
    """
    return Team(attach=eval_model_path(lconf, rconf, fold, "attach"),
                relate=eval_model_path(lconf, rconf, fold, "relate"))


def attelo_sent_model_paths(lconf, rconf, fold):
    """
    Return attelo intra/intersentential model paths
    """
    return Team(attach=eval_model_path(lconf, rconf, fold, "sent-attach"),
                relate=eval_model_path(lconf, rconf, fold, "sent-relate"))


_ATTELO_CONFIG_ARGS = ['--config', ATTELO_CONFIG_FILE]


def _attelo_model_args(lconf, rconf, fold, intra=False):
    """
    Return command line args for attelo model flags
    """
    if intra:
        paths = attelo_sent_model_paths(lconf, rconf, fold)
    else:
        paths = attelo_doc_model_paths(lconf, rconf, fold)
    return ["--attachment-model", paths.attach,
            "--relation-model", paths.relate]


def censor_flags(flags):
    """
    Return flags that are not in the list of things the harness
    handles by iteslf
    """
    return [f for f in flags if not is_intra(f)]


def intra_flags(flags):
    """
    Return flags that are not in the list of things the harness
    handles by iteslf
    """
    return [f for f in flags if is_intra(f)]


def is_intra(flag):
    """
    Return True if a flag corresponds to intra/intersential
    decoding
    """
    return isinstance(flag, IntraFlag)


def _attelo_fold_args(lconf, fold):
    """
    Return flags for picking out the attelo fold file (and fold
    number), if relevant
    """
    if fold is None:
        return []
    else:
        return ["--fold", str(fold),
                "--fold-file", lconf.fold_file]


class LearnArgs(CliArgs):
    """
    cmdline args for attelo learn
    """
    def __init__(self, lconf, rconf, fold, intra=False):
        super(LearnArgs, self).__init__()
        self.lconf = lconf
        self.rconf = rconf
        self.fold = fold
        self.intra = intra

    def parser(self):
        psr = argparse.ArgumentParser()
        att.learn.config_argparser(psr)
        return psr

    def argv(self):
        lconf = self.lconf
        rconf = self.rconf
        fold = self.fold

        args = []
        args.extend(_attelo_dpack_args(lconf))
        args.extend(_ATTELO_CONFIG_ARGS)
        args.extend(_attelo_model_args(lconf, rconf, fold,
                                       intra=self.intra))
        args.extend(_attelo_fold_args(lconf, fold))

        args.extend(["--learner", rconf.attach.name])
        args.extend(rconf.attach.flags)
        if rconf.relate is not None:
            args.extend(["--relation-learner", rconf.relate.name])
            # yuck: we assume that learner and relation learner flags
            # are compatible
            args.extend(rconf.relate.flags)
        decoder = rconf.attach.decoder
        if decoder is None and rconf.relate is not None:
            decoder = rconf.relate.decoder
        if decoder is not None:
            args.extend(["--decoder", decoder.name])
            # intercept fake intra-inter flag because we handle this on
            # the harness level
            args.extend(censor_flags(decoder.flags))

        if self.intra:
            args.extend(["--intrasentential"])
        return args


class DecodeArgs(CliArgs):
    """
    cmdline args for attelo decode
    """
    def __init__(self, lconf, econf, fold):
        super(DecodeArgs, self).__init__()
        self.lconf = lconf
        self.econf = econf
        self.fold = fold

    def parser(self):
        psr = argparse.ArgumentParser()
        att.decode.config_argparser(psr)
        return psr

    def argv(self):
        lconf = self.lconf
        econf = self.econf
        fold = self.fold

        args = []
        args.extend(_attelo_dpack_args(lconf))
        args.extend(_ATTELO_CONFIG_ARGS)
        args.extend(_attelo_model_args(lconf, econf.learner, fold))
        args.extend(_attelo_fold_args(lconf, fold))
        args.extend(["--decoder", econf.decoder.name,
                     "--output", decode_output_path(lconf, econf, fold)])
        args.extend(censor_flags(econf.decoder.flags))
        return args


class InspectArgs(CliArgs):
    "args for attelo inspect"
    def __init__(self, lconf, rconf, fold=None, intra=False):
        self.lconf = lconf
        self.rconf = rconf
        self.fold = fold
        self.intra = intra
        super(InspectArgs, self).__init__()

    def parser(self):
        """
        The argparser that would be called on context manager
        entry
        """
        psr = argparse.ArgumentParser()
        att.inspect.config_argparser(psr)
        return psr

    def argv(self):
        """
        Command line arguments that would correspond to this
        configuration

        :rtype: `[String]`
        """
        lconf = self.lconf
        rconf = self.rconf
        info_output = model_info_path(lconf, rconf, self.fold, self.intra)
        argv = [features_path(lconf),
                vocab_path(lconf),
                '--output', info_output]
        argv.extend(_attelo_model_args(lconf, rconf, self.fold, self.intra))
        return argv
# pylint: enable=too-many-instance-attributes


class GoldGraphArgs(CliArgs):
    'cmd line args to generate graphs (gold set)'
    def __init__(self, lconf):
        self.lconf = lconf
        super(GoldGraphArgs, self).__init__()

    def parser(self):
        psr = argparse.ArgumentParser()
        att.graph.config_argparser(psr)
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
        att.graph.config_argparser(psr)
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
