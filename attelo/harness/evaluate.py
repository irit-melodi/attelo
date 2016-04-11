# Author: Eric Kow
# License: CeCILL-B (French BSD3-like)

"""
Default harness evaluation.

The bulk of your test harness logic can often consist
of just calling these functions, but if you want to
customise how your harnesses run, beyond the settings
that are offered in the `Harness` interface, you could
just implement variants on these functions instead.
"""

from __future__ import print_function
from os import path as fp
import glob
import os
import sys

from joblib import Parallel

from attelo.io import (load_multipack,
                       load_fold_dict)
from attelo.harness.util import (call, force_symlink, timestamp)

from .config import (ClusterStage, DataConfig)
from .parse import (decode_on_the_fly,
                    delayed_decode,
                    learn,
                    post_decode)
from .report import (mk_fold_report,
                     mk_global_report,
                     mk_test_report)

# pylint: disable=too-few-public-methods


# ---------------------------------------------------------------------
# CODE CONVENTIONS USED HERE
# ---------------------------------------------------------------------
#
# hconf - harness config :: HarnessConfig
# rconf - learner config :: LearnerConfig
# econf - evaluation config :: EvaluationConfig
# dconf - data config :: DataConfig

# ---------------------------------------------------------------------
# user feedback
# ---------------------------------------------------------------------


def _corpus_banner(hconf):
    "banner to announce the corpus"
    return "\n".join(["==========" * 7,
                      hconf.dataset,
                      "==========" * 7])


def _fold_banner(hconf, fold):
    "banner to announce the next fold"
    return "\n".join(["==========" * 6,
                      "fold %d [%s]" % (fold, hconf.dataset),
                      "==========" * 6])

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
        data_file = fp.join(data_dir, fname)
        eval_file = fp.join(eval_dir, fname)
        if fp.isfile(data_file) and not fp.exists(eval_file):
            os.link(data_file, eval_file)


def _link_model_files(old_dir, new_dir):
    """
    Hardlink any fold-level or combined folds files
    """
    for old_mpath in glob.glob(fp.join(old_dir, '*', '*model*')):
        old_fold_dir_bn = fp.basename(fp.dirname(old_mpath))
        new_fold_dir = fp.join(new_dir, old_fold_dir_bn)
        new_mpath = fp.join(new_fold_dir, fp.basename(old_mpath))
        if not fp.exists(new_fold_dir):
            os.makedirs(new_fold_dir)
        os.link(old_mpath, new_mpath)


def _link_fold_files(old_dir, new_dir):
    """
    Hardlink the fold file
    """
    for old_path in glob.glob(fp.join(old_dir, 'folds*.json')):
        new_path = fp.join(new_dir, fp.basename(old_path))
        os.link(old_path, new_path)


def _create_tstamped_dir(prefix, suffix):
    """
    Given a path prefix (eg. 'foo/bar') and a new suffix
    (eg. quux),

    If the desired path (eg. 'foo/bar-quux') already exists,
    return False.
    Otherwise:

    1. Create a directory at the desired path
    2. Rename any existing prefix-'current' link
       to prefix-'previous'
    3. Link prefix-suffix to prefix-'current'
    4. Return True
    """
    old = prefix + '-previous'
    new = prefix + '-current'
    actual_new = prefix + '-' + suffix
    if fp.exists(actual_new):
        return False
    else:
        os.makedirs(actual_new)
        if fp.exists(new):
            actual_old = fp.realpath(prefix + '-current')
            force_symlink(fp.basename(actual_old), old)
        force_symlink(fp.basename(actual_new), new)
        return True


def prepare_dirs(runcfg, data_dir):
    """
    Return eval and scratch directory paths
    """
    eval_prefix = fp.join(data_dir, "eval")
    scratch_prefix = fp.join(data_dir, "scratch")

    eval_current = eval_prefix + '-current'
    scratch_current = scratch_prefix + '-current'
    stage = runcfg.stage

    if (runcfg.mode == 'resume' or stage in [ClusterStage.main,
                                             ClusterStage.combined_models,
                                             ClusterStage.end]):
        if not fp.exists(eval_current) or not fp.exists(scratch_current):
            sys.exit("No currently running evaluation to resume!")
        else:
            eval_dir = fp.realpath(eval_current)
            scratch_dir = fp.realpath(scratch_current)
            # in case there are any new data files to link
            _link_data_files(data_dir, eval_dir)
            return eval_dir, scratch_dir
    else:
        eval_actual_old = fp.realpath(eval_current)
        scratch_actual_old = fp.realpath(scratch_current)
        tstamp = timestamp()
        if _create_tstamped_dir(eval_prefix, tstamp):
            eval_dir = eval_prefix + '-' + tstamp
            scratch_dir = scratch_prefix + '-' + tstamp
            _create_tstamped_dir(scratch_prefix, tstamp)
            _link_data_files(data_dir, eval_dir)
            if runcfg.stage == 'jumpstart':
                _link_fold_files(eval_actual_old, eval_dir)
                _link_model_files(scratch_actual_old, scratch_dir)
        else:
            sys.exit("Try again in one minute")

        with open(fp.join(eval_dir, "versions-evaluate.txt"), "w") as stream:
            call(["pip", "freeze"], stdout=stream)

        return eval_dir, scratch_dir

# ---------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------


def do_fold(hconf, dconf, fold):
    """
    Run all learner/decoder combos within this fold
    """
    fold_dir = hconf.fold_dir_path(fold)
    print(_fold_banner(hconf, fold), file=sys.stderr)
    if not fp.exists(fold_dir):
        os.makedirs(fold_dir)

    # learn/decode for all models
    decoder_jobs = decode_on_the_fly(hconf, dconf, fold)
    Parallel(n_jobs=hconf.runcfg.n_jobs, verbose=True)(decoder_jobs)
    for econf in hconf.evaluations:
        post_decode(hconf, dconf, econf, fold)
    mk_fold_report(hconf, dconf, fold)


def do_global_decode(hconf, dconf):
    """
    Run decoder on test data (if available)
    """
    econf = hconf.test_evaluation
    if econf is not None:
        decoder_jobs = delayed_decode(hconf, dconf, econf, None)
        Parallel(n_jobs=hconf.runcfg.n_jobs, verbose=True)(decoder_jobs)
        post_decode(hconf, dconf, econf, None)


def _load_harness_multipack(hconf, test_data=False):
    """
    Load the multipack for our current configuration.

    Load the stripped features file if we don't actually need to
    use the features (this would only make sense on the cluster
    where evaluation is broken up into separate stages that we
    can fire on different nodes)

    Parameters
    ----------
    test_data: bool

    Returns
    -------
    mpack: Multipack
    """
    stripped_paths = hconf.mpack_paths(test_data, stripped=True)
    if (hconf.runcfg.stage in [ClusterStage.end, ClusterStage.start] and
        fp.exists(stripped_paths[2])):
        paths = stripped_paths
    else:
        paths = hconf.mpack_paths(test_data, stripped=False)
    mpack = load_multipack(paths[0],
                           paths[1],
                           paths[2],
                           paths[3],
                           corpus_path=(paths[4] if len(paths) == 5
                                        else None),  # WIP
                           verbose=True)
    return mpack


def _init_corpus(hconf):
    """Start evaluation; generate folds if needed

    :rtype: DataConfig or None
    """
    can_skip_folds = fp.exists(hconf.fold_file)
    msg_skip_folds = ('Skipping generation of fold files '
                      '(must have been jumpstarted)')

    if hconf.runcfg.stage is None:
        # standalone: we always have to load the datapack
        # because we'll need it for further stages
        mpack = _load_harness_multipack(hconf)
        if can_skip_folds:
            print(msg_skip_folds, file=sys.stderr)
            fold_dict = load_fold_dict(hconf.fold_file)
        else:
            fold_dict = hconf.create_folds(mpack)
        return DataConfig(pack=mpack, folds=fold_dict)
    elif hconf.runcfg.stage == ClusterStage.start:
        if can_skip_folds:
            # if we are just running --start and the fold file already
            # exists we can even bail out before reading the datapacks
            # because that's all we wanted them for
            print(msg_skip_folds, file=sys.stderr)
        else:
            mpack = _load_harness_multipack(hconf)
            hconf.create_folds(mpack)
        return None
    else:
        # any other stage: fold files have already been
        # created so we just read them in
        return DataConfig(pack=_load_harness_multipack(hconf),
                          folds=load_fold_dict(hconf.fold_file))


def evaluate_corpus(hconf):
    "Run evaluation on a corpus"
    print(_corpus_banner(hconf), file=sys.stderr)

    dconf = _init_corpus(hconf)
    if hconf.runcfg.stage in [None, ClusterStage.main]:
        foldset = hconf.runcfg.folds if hconf.runcfg.folds is not None\
            else frozenset(dconf.folds.values())
        for fold in foldset:
            do_fold(hconf, dconf, fold)

    if hconf.runcfg.stage in [None, ClusterStage.combined_models]:
        for econf in hconf.evaluations:
            learn(hconf, econf, dconf, None)
        if hconf.test_evaluation is not None:
            test_pack = _load_harness_multipack(hconf, test_data=True)
            test_dconf = DataConfig(pack=test_pack, folds=None)
            do_global_decode(hconf, test_dconf)
            mk_test_report(hconf, test_dconf)

    if hconf.runcfg.stage in [None, ClusterStage.end]:
        mk_global_report(hconf, dconf)
