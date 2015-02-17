# Author: Eric Kow
# License: CeCILL-B (French BSD3-like)

"""
run an experiment
"""

from __future__ import print_function
from os import path as fp
from collections import namedtuple
from enum import Enum
import argparse
import json
import os
import shutil
import sys

from joblib import (Parallel, delayed)

from attelo.args import args_to_decoder
from attelo.io import (load_data_pack, Torpor)
from attelo.harness.config import CliArgs
from attelo.harness.report import (mk_index)
from attelo.harness.util import\
    timestamp, call, force_symlink
import attelo.cmd as att

from ..local import (LEARNERS,
                     EVALUATIONS,
                     TRAINING_CORPORA,
                     ATTELO_CONFIG_FILE)
from ..util import (concat_i, latest_tmp)

# pylint: disable=too-few-public-methods

NAME = 'evaluate'
_DEBUG = 0


# pylint: disable=pointless-string-statement
LoopConfig = namedtuple("LoopConfig",
                        ["eval_dir",
                         "scratch_dir",
                         "stage",
                         "folds",
                         "fold_file",
                         "dataset"])
"that which is common to outerish loops"


DataConfig = namedtuple("DataConfig",
                        "pack folds")
"data tables we have read"
# pylint: enable=pointless-string-statement


class ClusterStage(Enum):
    '''
    What stage of cluster usage we are at
    '''
    start = 1
    main = 2
    combined_models = 3
    end = 4


# ---------------------------------------------------------------------
# CODE CONVENTIONS USED HERE
# ---------------------------------------------------------------------
#
# lconf - loop config :: LoopConfig
# rconf - learner config :: LearnerConfig
# econf - evaluation config :: EvaluationConfig
# dconf - data config :: DataConfig

# ---------------------------------------------------------------------
# user feedback
# ---------------------------------------------------------------------


def _exit_ungathered():
    """
    You don't seem to have run the gather command
    """
    sys.exit("""No data to run experiments on.
Please run `irit-rst-dt gather`""")


def _eval_banner(econf, lconf, fold):
    """
    Which combo of eval parameters are we running now?
    """
    return "\n".join(["----------" * 3,
                      "fold %d [%s]" % (fold, lconf.dataset),
                      "learner(s): %s" % econf.learner.key,
                      "decoder: %s" % econf.decoder.key,
                      "----------" * 3])


def _corpus_banner(lconf):
    "banner to announce the corpus"
    return "\n".join(["==========" * 7,
                      lconf.dataset,
                      "==========" * 7])


def _fold_banner(lconf, fold):
    "banner to announce the next fold"
    return "\n".join(["==========" * 6,
                      "fold %d [%s]" % (fold, lconf.dataset),
                      "==========" * 6])

# ---------------------------------------------------------------------
# attelo config
# ---------------------------------------------------------------------


def _attelo_model_args(lconf, rconf, fold):
    """
    Return command line args for attelo model flags
    """
    return ["--attachment-model",
            _eval_model_path(lconf, rconf, fold, "attach"),
            "--relation-model",
            _eval_model_path(lconf, rconf, fold, "relate")]


_ATTELO_CONFIG_ARGS = ['--config', ATTELO_CONFIG_FILE]


# pylint: disable=too-many-instance-attributes
class FakeEvalArgs(CliArgs):
    """
    Fake argparse object (to be subclassed)
    Things in common between attelo learn/decode

    Note: must be used as a context manager
    """
    def __init__(self, lconf, rconf, fold):
        self.lconf = lconf
        self.rconf = rconf
        self.fold = fold
        super(FakeEvalArgs, self).__init__()

    def parser(self):
        """
        The argparser that would be called on context manager
        entry
        """
        psr = argparse.ArgumentParser()
        att.enfold.config_argparser(psr)

    def argv(self):
        """
        Command line arguments that would correspond to this
        configuration

        :rtype: `[String]`
        """
        rconf = self.rconf
        lconf = self.lconf
        fold = self.fold

        argv = [_edu_input_path(lconf),
                _pairings_path(lconf),
                _features_path(lconf)]

        argv.extend(_ATTELO_CONFIG_ARGS)
        argv.extend(_attelo_model_args(lconf, rconf, fold))

        if fold is not None:
            argv.extend(["--fold", str(fold),
                         "--fold-file", lconf.fold_file])

        return argv

    # pylint: disable=no-member
    def __exit__(self, ctype, value, traceback):
        "Tidy up any open file handles, etc"
        if self.fold_file is not None:
            self.fold_file.close()
        super(FakeEvalArgs, self).__exit__(ctype, value, traceback)
    # pylint: enable=no-member


class FakeEnfoldArgs(CliArgs):
    """
    Fake argparse object that would be generated by attelo enfold
    """
    def __init__(self, lconf):
        self.lconf = lconf
        super(FakeEnfoldArgs, self).__init__()

    def parser(self):
        psr = argparse.ArgumentParser()
        att.enfold.config_argparser(psr)
        return psr

    def argv(self):
        """
        Command line arguments that would correspond to this
        configuration

        :rtype: `[String]`
        """
        lconf = self.lconf
        args = [_edu_input_path(lconf),
                _pairings_path(lconf),
                _features_path(lconf),
                "--config", ATTELO_CONFIG_FILE,
                "--output", lconf.fold_file]
        return args

    # pylint: disable=no-member
    def __exit__(self, ctype, value, traceback):
        "Tidy up any open file handles, etc"
        self.output.close()
        super(FakeEnfoldArgs, self).__exit__(ctype, value, traceback)
    # pylint: enable=no-member


class FakeLearnArgs(FakeEvalArgs):
    """
    Fake argparse object that would be generated by attelo learn.
    """
    def __init__(self, lconf, rconf, fold):
        super(FakeLearnArgs, self).__init__(lconf, rconf, fold)

    def parser(self):
        psr = argparse.ArgumentParser()
        att.learn.config_argparser(psr)
        return psr

    def argv(self):
        rconf = self.rconf
        args = super(FakeLearnArgs, self).argv()
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
            args.extend(decoder.flags)
        return args


class FakeDecodeArgs(FakeEvalArgs):
    """
    Fake argparse object that would be generated by attelo decode
    """
    def __init__(self, lconf, econf, fold):
        super(FakeDecodeArgs, self).__init__(lconf, econf.learner, fold)
        self.econf = econf

    def parser(self):
        psr = argparse.ArgumentParser()
        att.decode.config_argparser(psr)
        return psr

    def argv(self):
        lconf = self.lconf
        econf = self.econf
        fold = self.fold
        args = super(FakeDecodeArgs, self).argv()
        args.extend(["--decoder", econf.decoder.name,
                     "--output", _decode_output_path(lconf, econf, fold)])
        args.extend(econf.decoder.flags)
        return args


class FakeReportArgs(CliArgs):
    "args for attelo report"
    def __init__(self, lconf, fold):
        self.lconf = lconf
        self.fold = fold
        super(FakeReportArgs, self).__init__()

    def parser(self):
        """
        The argparser that would be called on context manager
        entry
        """
        psr = argparse.ArgumentParser()
        att.report.config_argparser(psr)
        return psr

    def argv(self):
        """
        Command line arguments that would correspond to this
        configuration

        :rtype: `[String]`
        """
        lconf = self.lconf
        index_path = fp.join(_report_parent_dir(lconf, self.fold),
                             'index.json')
        argv = [_edu_input_path(lconf),
                _pairings_path(lconf),
                _features_path(lconf),
                "--index", index_path,
                "--config", ATTELO_CONFIG_FILE,
                "--fold-file", lconf.fold_file,
                "--output", _report_dir(lconf, self.fold)]
        return argv

    # pylint: disable=no-member
    def __exit__(self, ctype, value, traceback):
        "Tidy up any open file handles, etc"
        self.fold_file.close()
        super(FakeReportArgs, self).__exit__(ctype, value, traceback)
    # pylint: enable=no-member
# pylint: enable=too-many-instance-attributes


class FakeInspectArgs(CliArgs):
    "args for attelo inspect"
    def __init__(self, lconf, rconf, fold=None):
        self.lconf = lconf
        self.rconf = rconf
        self.fold = fold
        super(FakeInspectArgs, self).__init__()

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
        argv = [_features_path(lconf),
                _vocab_path(lconf),
                '--output', _model_info_path(lconf, rconf, self.fold)]
        argv.extend(_attelo_model_args(lconf, rconf, self.fold))
        return argv

    # pylint: disable=no-member
    def __exit__(self, ctype, value, traceback):
        "Tidy up any open file handles, etc"
        super(FakeInspectArgs, self).__exit__(ctype, value, traceback)
    # pylint: enable=no-member
# pylint: enable=too-many-instance-attributes


class FakeGoldGraphArgs(CliArgs):
    'cmd line args to generate graphs (gold set)'
    def __init__(self, lconf):
        self.lconf = lconf
        super(FakeGoldGraphArgs, self).__init__()

    def parser(self):
        psr = argparse.ArgumentParser()
        att.graph.config_argparser(psr)
        return psr

    def argv(self):
        lconf = self.lconf
        has_stripped = fp.exists(_features_path(lconf, stripped=True))
        argv = [_edu_input_path(lconf),
                '--quiet',
                '--gold',
                _pairings_path(lconf),
                _features_path(lconf, stripped=has_stripped),
                '--output',
                fp.join(_report_dir(lconf, None),
                        'graphs-gold')]
        return argv


class FakeGraphArgs(CliArgs):
    'cmd line args to generate graphs (for a fold)'
    def __init__(self, lconf, econf, fold):
        self.lconf = lconf
        self.econf = econf
        self.fold = fold
        super(FakeGraphArgs, self).__init__()

    def parser(self):
        psr = argparse.ArgumentParser()
        att.graph.config_argparser(psr)
        return psr

    def argv(self):
        lconf = self.lconf
        econf = self.econf
        fold = self.fold
        output_path = fp.join(_report_dir(lconf, None),
                              'graphs-' + _fold_dir_basename(fold),
                              econf.key)
        argv = [_edu_input_path(lconf),
                '--predictions',
                _decode_output_path(lconf, econf, fold),
                '--graphviz-timeout', str(15),
                '--quiet',
                '--output', output_path]
        return argv

# ---------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------


def _eval_data_path(lconf, ext):
    """
    Path to data file in the evaluation dir
    """
    return os.path.join(lconf.eval_dir,
                        "%s.%s" % (lconf.dataset, ext))


def _features_path(lconf, stripped=False):
    """
    Path to the feature file in the evaluation dir
    """
    ext = 'relations.sparse'
    if stripped:
        ext += '.stripped'
    return _eval_data_path(lconf, ext)


def _vocab_path(lconf):
    """
    Path to the vocab file in the evaluation dir
    """
    return _features_path(lconf) + '.vocab'


def _edu_input_path(lconf):
    """
    Path to the feature file in the evaluation dir
    """
    return _features_path(lconf) + '.edu_input'


def _pairings_path(lconf):
    """
    Path to the pairings file in the evaluation dir
    """
    return _features_path(lconf) + '.pairings'


def _fold_dir_basename(fold):
    "Relative directory for working within a given fold"
    return "fold-%d" % fold


def _fold_dir_path(lconf, fold):
    "Scratch directory for working within a given fold"
    return os.path.join(lconf.scratch_dir,
                        _fold_dir_basename(fold))


def _combined_dir_path(lconf):
    "Scratch directory for working within the global config"
    return fp.join(lconf.scratch_dir, 'combined')


def _model_basename(lconf, rconf, mtype, ext):
    "Basic filename for a model"
    template = '{dataset}.{learner}.{task}.{ext}'
    return template.format(dataset=lconf.dataset,
                           learner=rconf.key,
                           task=mtype,
                           ext=ext)


def _eval_model_path(lconf, rconf, fold, mtype):
    "Model for a given loop/eval config and fold"
    parent_dir = _combined_dir_path(lconf) if fold is None\
        else _fold_dir_path(lconf, fold)
    return fp.join(parent_dir,
                   _model_basename(lconf, rconf, mtype, 'model'))


def _decode_output_basename(econf):
    "Model for a given loop/eval config and fold"
    return ".".join(["output", econf.key])


def _decode_output_path(lconf, econf, fold):
    "Model for a given loop/eval config and fold"
    fold_dir = _fold_dir_path(lconf, fold)
    return os.path.join(fold_dir, _decode_output_basename(econf))


def _report_dir_basename(lconf):
    "Relative directory for a report directory"
    return "reports-%s" % lconf.dataset


def _report_parent_dir(lconf, fold=None):
    "Directory that a report dir would be placed in"
    if fold is None:
        return lconf.scratch_dir
    else:
        return _fold_dir_path(lconf, fold)


def _report_dir(lconf, fold=None):
    """
    Path to a score file given a parent dir.
    You'll need to tack an extension onto this
    """
    return fp.join(_report_parent_dir(lconf, fold),
                   _report_dir_basename(lconf))


def _model_info_path(lconf, rconf, fold=None):
    """
    Path to the model output file
    """
    template = "discr-features.{learner}.txt"
    return fp.join(_report_dir(lconf, fold),
                   template.format(learner=rconf.key))


# ---------------------------------------------------------------------
# preparation
# ---------------------------------------------------------------------

def _link_data_files(data_dir, eval_dir):
    """
    Hard-link all files from the data dir into the evaluation
    directory. This does not cost space and it makes future
    archiving a bit more straightforward
    """
    for fname in os.listdir(data_dir):
        data_file = os.path.join(data_dir, fname)
        eval_file = os.path.join(eval_dir, fname)
        if os.path.isfile(data_file):
            os.link(data_file, eval_file)


def _create_eval_dirs(args, data_dir):
    """
    Return eval and scatch directory paths
    """

    eval_current = fp.join(data_dir, "eval-current")
    scratch_current = fp.join(data_dir, "scratch-current")
    stage = args_to_stage(args)

    if args.resume or stage in [ClusterStage.main,
                                ClusterStage.combined_models,
                                ClusterStage.end]:
        if not fp.exists(eval_current) or not fp.exists(scratch_current):
            sys.exit("No currently running evaluation to resume!")
        else:
            return eval_current, scratch_current
    else:
        tstamp = "TEST" if _DEBUG else timestamp()
        eval_dir = fp.join(data_dir, "eval-" + tstamp)
        if not fp.exists(eval_dir):
            os.makedirs(eval_dir)
            _link_data_files(data_dir, eval_dir)
            force_symlink(fp.basename(eval_dir), eval_current)
        elif not _DEBUG:
            sys.exit("Try again in literally one second")

        scratch_dir = fp.join(data_dir, "scratch-" + tstamp)
        if not fp.exists(scratch_dir):
            os.makedirs(scratch_dir)
            force_symlink(fp.basename(scratch_dir), scratch_current)

        return eval_dir, scratch_dir

# ---------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------


def _delayed_learn(lconf, dconf, rconf, fold):
    """
    Return possible futures for learning models for this
    fold
    """
    fold_dir = _fold_dir_path(lconf, fold)
    if not os.path.exists(fold_dir):
        os.makedirs(fold_dir)

    with FakeLearnArgs(lconf, rconf, fold) as args:
        if fp.exists(args.attachment_model) and fp.exists(args.relation_model):
            print("reusing %s model (already built)" % rconf.key,
                  file=sys.stderr)
            return []
        subpack = dconf.pack.training(dconf.folds, fold)
        return att.learn.delayed_main_for_harness(args, subpack)


def _delayed_learn_combined(lconf, dconf, rconf):
    """
    Return possible futures for learning models with the
    given learner configuration
    """
    combined_dir = _combined_dir_path(lconf)
    if not os.path.exists(combined_dir):
        os.makedirs(combined_dir)

    with FakeLearnArgs(lconf, rconf, None) as args:
        if fp.exists(args.attachment_model) and fp.exists(args.relation_model):
            print("reusing %s model (already built)" % rconf.key,
                  file=sys.stderr)
            return []
        return att.learn.delayed_main_for_harness(args, dconf.pack)


def _say_if_decoded(lconf, econf, fold):
    """
    If we have already done the decoding for a given config
    and fold, say so and return True
    """
    if fp.exists(_decode_output_path(lconf, econf, fold)):
        print("skipping %s/%s (already done)" % (econf.learner.key,
                                                 econf.decoder.key),
              file=sys.stderr)
        return True
    else:
        return False


def _delayed_decode(lconf, dconf, econf, fold):
    """
    Return possible futures for decoding groups within
    this model/decoder combo for the given fold
    """
    if _say_if_decoded(lconf, econf, fold):
        return []

    fold_dir = _fold_dir_path(lconf, fold)
    if not os.path.exists(fold_dir):
        os.makedirs(fold_dir)
    with FakeDecodeArgs(lconf, econf, fold) as args:
        decoder = args_to_decoder(args)
        subpack = dconf.pack.testing(dconf.folds, fold)
        models = att.decode.load_models(args)
        return att.decode.delayed_main_for_harness(args, decoder,
                                                   subpack, models)


def _post_decode(lconf, dconf, econf, fold):
    """
    Join together output files from this model/decoder combo
    """
    if _say_if_decoded(lconf, econf, fold):
        return

    print(_eval_banner(econf, lconf, fold), file=sys.stderr)
    with FakeDecodeArgs(lconf, econf, fold) as args:
        subpack = dconf.pack.testing(dconf.folds, fold)
        att.decode.concatenate_outputs(args, subpack)


def _generate_fold_file(lconf, dpack):
    """
    Generate the folds file
    """
    with FakeEnfoldArgs(lconf) as args:
        att.enfold.main_for_harness(args, dpack)


def _mk_report(args, index, dconf):
    "helper for report generation"
    with open(args.index, 'w') as ostream:
        json.dump(index, ostream)
    att.report.main_for_harness(args, dconf.pack, args.output)
    for rconf in LEARNERS:
        with FakeInspectArgs(args.lconf, rconf, args.fold) as inspect_args:
            _mk_model_summary(inspect_args)


def _mk_model_summary(args):
    "generate summary of best model features"
    att.inspect.main_for_harness(args)


def _mk_fold_report(lconf, dconf, fold):
    "Generate reports for scores"
    configurations = [(econf, _decode_output_basename(econf))
                      for econf in EVALUATIONS]
    index = mk_index([(fold, '.')], configurations)
    with FakeReportArgs(lconf, fold) as args:
        _mk_report(args, index, dconf)


def _mk_econf_graphs(lconf, econf, fold):
    "Generate graphs for a single configuration"
    with FakeGraphArgs(lconf, econf, fold) as args:
        att.graph.main_for_harness(args)


def _mk_graphs(lconf, dconf):
    "Generate graphs for the gold data and for one of the folds"
    with FakeGoldGraphArgs(lconf) as args:
        if fp.exists(args.output):
            print("skipping gold graphs (already done)",
                  file=sys.stderr)
        else:
            with Torpor('creating gold graphs'):
                att.graph.main_for_harness(args)
    fold = sorted(set(dconf.folds.values()))[0]

    with Torpor('creating graphs for fold {}'.format(fold),
                sameline=False):
        jobs = [delayed(_mk_econf_graphs)(lconf, econf, fold)
                for econf in EVALUATIONS]
        Parallel(n_jobs=-1, verbose=5)(jobs)


def _mk_global_report(lconf, dconf):
    "Generate reports for all folds"
    folds = [(f, _fold_dir_basename(f))
             for f in frozenset(dconf.folds.values())]
    configurations = [(econf, _decode_output_basename(econf))
                      for econf in EVALUATIONS]
    index = mk_index(folds, configurations)
    final_report_dir = fp.join(lconf.eval_dir,
                               _report_dir_basename(lconf))
    with FakeReportArgs(lconf, None) as args:
        _mk_report(args, index, dconf)
        # _mk_graphs(lconf, dconf)
        if fp.exists(final_report_dir):
            shutil.rmtree(final_report_dir)
        shutil.copytree(args.output, final_report_dir)
    # this can happen if resuming a report; better copy
    # it again
    print('Report saved in ', final_report_dir,
          file=sys.stderr)


def _do_fold(lconf, dconf, fold):
    """
    Run all learner/decoder combos within this fold
    """
    fold_dir = _fold_dir_path(lconf, fold)
    print(_fold_banner(lconf, fold), file=sys.stderr)
    if not os.path.exists(fold_dir):
        os.makedirs(fold_dir)

    # learn all models in parallel
    learner_jobs = concat_i(_delayed_learn(lconf, dconf, rconf, fold)
                            for rconf in LEARNERS)
    Parallel(n_jobs=-1, verbose=5)(learner_jobs)
    # run all model/decoder joblets in parallel
    decoder_jobs = concat_i(_delayed_decode(lconf, dconf, econf, fold)
                            for econf in EVALUATIONS)
    Parallel(n_jobs=-1, verbose=5)(decoder_jobs)
    for econf in EVALUATIONS:
        _post_decode(lconf, dconf, econf, fold)
    fold_dir = _fold_dir_path(lconf, fold)
    _mk_fold_report(lconf, dconf, fold)


def _mk_combined_models(lconf, dconf):
    """
    Create global for all learners
    """
    jobs = concat_i(_delayed_learn_combined(lconf, dconf, learner)
                    for learner in LEARNERS)
    Parallel(n_jobs=-1, verbose=5)(jobs)


def _is_standalone_or(lconf, stage):
    """
    True if we are in standalone mode (do everything)
    or in a given cluster stage
    """
    return lconf.stage is None or lconf.stage == stage


def _do_corpus(lconf):
    "Run evaluation on a corpus"
    print(_corpus_banner(lconf), file=sys.stderr)

    edus_file = _edu_input_path(lconf)
    if not os.path.exists(edus_file):
        _exit_ungathered()

    has_stripped = (lconf.stage == ClusterStage.end
                    and fp.exists(_features_path(lconf, stripped=True)))
    dpack = load_data_pack(edus_file,
                           _pairings_path(lconf),
                           _features_path(lconf, stripped=has_stripped),
                           verbose=True)

    _generate_fold_file(lconf, dpack)

    with open(lconf.fold_file) as f_in:
        dconf = DataConfig(pack=dpack,
                           folds=json.load(f_in))

    if _is_standalone_or(lconf, ClusterStage.main):
        foldset = lconf.folds if lconf.folds is not None\
            else frozenset(dconf.folds.values())
        for fold in foldset:
            _do_fold(lconf, dconf, fold)

    if _is_standalone_or(lconf, ClusterStage.combined_models):
        _mk_combined_models(lconf, dconf)

    if _is_standalone_or(lconf, ClusterStage.end):
        _mk_global_report(lconf, dconf)

# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------


def config_argparser(psr):
    """
    Subcommand flags.

    You should create and pass in the subparser to which the flags
    are to be added.
    """
    psr.set_defaults(func=main)
    psr.add_argument("--resume",
                     default=False, action="store_true",
                     help="resume previous interrupted evaluation")
    cluster_grp = psr.add_mutually_exclusive_group()
    cluster_grp.add_argument("--start", action='store_true',
                             default=False,
                             help="initialise an evaluation but don't run it "
                             "(cluster mode)")
    cluster_grp.add_argument("--folds", metavar='N', type=int, nargs='+',
                             help="run only these folds (cluster mode)")
    cluster_grp.add_argument("--combined-models", action='store_true',
                             help="generate only the combined model")
    cluster_grp.add_argument("--end", action='store_true',
                             default=False,
                             help="generate report only (cluster mode)")


def args_to_stage(args):
    "return the cluster stage from the CLI args"

    if args.start:
        return ClusterStage.start
    elif args.folds is not None:
        return ClusterStage.main
    elif args.combined_models:
        return ClusterStage.combined_models
    elif args.end:
        return ClusterStage.end
    else:
        return None


def main(args):
    """
    Subcommand main.

    You shouldn't need to call this yourself if you're using
    `config_argparser`
    """
    sys.setrecursionlimit(10000)
    stage = args_to_stage(args)
    data_dir = latest_tmp()
    if not os.path.exists(data_dir):
        _exit_ungathered()
    eval_dir, scratch_dir = _create_eval_dirs(args, data_dir)

    with open(os.path.join(eval_dir, "versions.txt"), "w") as stream:
        call(["pip", "freeze"], stdout=stream)

    if stage == ClusterStage.start:
        # all done! just wanted to create the directory
        return

    for corpus in TRAINING_CORPORA:
        dataset = os.path.basename(corpus)
        fold_file = os.path.join(eval_dir,
                                 "folds-%s.json" % dataset)

        lconf = LoopConfig(eval_dir=eval_dir,
                           scratch_dir=scratch_dir,
                           folds=args.folds,
                           stage=stage,
                           fold_file=fold_file,
                           dataset=dataset)
        _do_corpus(lconf)
