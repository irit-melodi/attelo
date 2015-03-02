# Author: Eric Kow
# License: CeCILL-B (French BSD3-like)

"""
run an experiment
"""

from __future__ import print_function
from collections import Counter
from os import path as fp
import codecs
import itertools as itr
import glob
import os
import shutil
import sys

from joblib import (Parallel, delayed)

from attelo.io import (load_data_pack, load_predictions,
                       load_fold_dict, save_fold_dict,
                       load_model, load_vocab)
from attelo.decoding.intra import (IntraInterPair)
from attelo.harness.report import (Slice, full_report)
from attelo.harness.util import\
    timestamp, call, force_symlink
from attelo.learning import (Task)
from attelo.table import (for_intra)
from attelo.util import (Team, mk_rng)
import attelo.harness as ath
import attelo.fold
import attelo.score
import attelo.report

from ..graph import (mk_graphs)
from ..local import (EVALUATIONS,
                     DETAILED_EVALUATIONS,
                     TRAINING_CORPORA)
from ..path import (attelo_doc_model_paths,
                    attelo_sent_model_paths,
                    combined_dir_path,
                    decode_output_path,
                    edu_input_path,
                    eval_model_path,
                    features_path,
                    fold_dir_path,
                    model_info_path,
                    pairings_path,
                    report_dir_basename,
                    report_dir_path,
                    vocab_path)
from ..util import (concat_i,
                    latest_tmp,
                    md5sum_file)
from ..loop import (LoopConfig,
                    DataConfig,
                    ClusterStage)

# pylint: disable=too-few-public-methods

NAME = 'evaluate'
_DEBUG = 0

LEARNERS = {e.learner.key: e.learner for e in EVALUATIONS}.values()

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
    msg = ("Reassembling "
           "fold {fnum} [{dset}]\t"
           "learner(s): {learner}\t"
           "decoder: {decoder}")
    return msg.format(fnum=fold,
                      dset=lconf.dataset,
                      learner=econf.learner.key,
                      decoder=econf.decoder.key)


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


def _create_eval_dirs(args, data_dir, jumpstart):
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
            sys.exit("Try again in one minute")

        scratch_dir = fp.join(data_dir, "scratch-" + tstamp)
        if not fp.exists(scratch_dir):
            os.makedirs(scratch_dir)
            if jumpstart:
                _link_model_files(scratch_current, scratch_dir)
            force_symlink(fp.basename(scratch_dir), scratch_current)

        with open(fp.join(eval_dir, "versions-evaluate.txt"), "w") as stream:
            call(["pip", "freeze"], stdout=stream)

        return eval_dir, scratch_dir


def _sanity_check_config():
    """
    Die if there's anything odd about the config
    """
    conf_counts = Counter(econf.key for econf in EVALUATIONS)
    bad_confs = [k for k, v in conf_counts.items() if v > 1]
    if bad_confs:
        oops = ("Sorry, there's an error in your configuration.\n"
                "I don't dare to start evaluation until you fix it.\n"
                "ERROR! -----------------vvvv---------------------\n"
                "The following configurations more than once:{}\n"
                "ERROR! -----------------^^^^^--------------------"
                "").format("\n".join(bad_confs))
        sys.exit(oops)

# ---------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------


def _parallel(lconf, n_jobs=None, verbose=None):
    """
    Run some delayed jobs in parallel (or sequentially
    depending on our settings)
    """
    n_jobs = n_jobs or lconf.n_jobs
    verbose = verbose or 5

    def sequential(jobs):
        """
        run jobs in truly sequential fashion without any of
        this parallel nonsense
        """
        # pylint: disable=star-args
        for func, args, kwargs in jobs:
            func(*args, **kwargs)
        # pylint: enable=star-args

    if n_jobs == 0:
        return sequential
    else:
        return Parallel(n_jobs=n_jobs, verbose=verbose)


def _get_learn_job(lconf, rconf, subpack, paths, task):
    'learn a model and write it to the given output path'

    if task == Task.attach:
        sub_rconf = rconf.attach
        output_path = paths.attach
    elif task == Task.relate:
        sub_rconf = rconf.relate or rconf.attach
        output_path = paths.relate
    else:
        raise ValueError('Unknown learning task: {}'.format(task))

    if sub_rconf.key == 'oracle':
        return None
    elif fp.exists(output_path):
        print(("reusing {key} {task} model (already built): {path}"
               "").format(key=sub_rconf.key,
                          task=task.name,
                          path=fp.relpath(output_path, lconf.scratch_dir)),
              file=sys.stderr)
    else:
        learn_fn = ath.learn.learn
        learners = Team(attach=rconf.attach,
                        relate=rconf.relate or rconf.attach)
        learners = learners.fmap(lambda x: x.payload)
        return delayed(learn_fn)(subpack, learners, task, output_path,
                                 quiet=False)


def _delayed_learn(lconf, dconf, rconf, fold, include_intra):
    """
    Return possible futures for learning models for this
    fold
    """
    if fold is None:
        parent_dir = combined_dir_path(lconf)
        get_subpack = lambda d: d
    else:
        parent_dir = fold_dir_path(lconf, fold)
        get_subpack = lambda d: d.training(dconf.folds, fold)

    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    jobs = []
    if True:
        subpack = get_subpack(dconf.pack)
        paths = attelo_doc_model_paths(lconf, rconf, fold)
        jobs.append(_get_learn_job(lconf, rconf, subpack, paths, Task.attach))
        jobs.append(_get_learn_job(lconf, rconf, subpack, paths, Task.relate))
    if include_intra:
        subpack = for_intra(get_subpack(dconf.pack))
        paths = attelo_sent_model_paths(lconf, rconf, fold)
        jobs.append(_get_learn_job(lconf, rconf, subpack, paths, Task.attach))
        jobs.append(_get_learn_job(lconf, rconf, subpack, paths, Task.relate))
    return [j for j in jobs if j is not None]


def _say_if_decoded(lconf, econf, fold, stage='decoding'):
    """
    If we have already done the decoding for a given config
    and fold, say so and return True
    """
    if fp.exists(decode_output_path(lconf, econf, fold)):
        print(("skipping {stage} {learner} {decoder} "
               "(already done)").format(stage=stage,
                                        learner=econf.learner.key,
                                        decoder=econf.decoder.key),
              file=sys.stderr)
        return True
    else:
        return False


def _delayed_decode(lconf, dconf, econf, fold):
    """
    Return possible futures for decoding groups within
    this model/decoder combo for the given fold
    """
    if _say_if_decoded(lconf, econf, fold, stage='decoding'):
        return []

    fold_dir = fold_dir_path(lconf, fold)
    if not os.path.exists(fold_dir):
        os.makedirs(fold_dir)

    subpack = dconf.pack.testing(dconf.folds, fold)
    doc_model_paths = attelo_doc_model_paths(lconf, econf.learner, fold)
    intra_flag = econf.settings.intra
    if intra_flag is not None:
        sent_model_paths =\
            attelo_sent_model_paths(lconf, econf.learner, fold)

        intra_model = Team('oracle', 'oracle')\
            if intra_flag.intra_oracle\
            else sent_model_paths.fmap(load_model)
        inter_model = Team('oracle', 'oracle')\
            if intra_flag.inter_oracle\
            else doc_model_paths.fmap(load_model)

        models = IntraInterPair(intra=intra_model,
                                inter=inter_model)
    else:
        models = doc_model_paths.fmap(load_model)

    return ath.decode.jobs(subpack, models,
                           econf.decoder.payload,
                           econf.settings.mode,
                           decode_output_path(lconf, econf, fold))


def _post_decode(lconf, dconf, econf, fold):
    """
    Join together output files from this model/decoder combo
    """
    if _say_if_decoded(lconf, econf, fold, stage='reassembly'):
        return

    print(_eval_banner(econf, lconf, fold), file=sys.stderr)
    subpack = dconf.pack.testing(dconf.folds, fold)
    ath.decode.concatenate_outputs(subpack,
                                   decode_output_path(lconf, econf, fold))


def _generate_fold_file(lconf, dpack):
    """
    Generate the folds file
    """
    rng = mk_rng()
    fold_dict = attelo.fold.make_n_fold(dpack, 10, rng)
    save_fold_dict(fold_dict, lconf.fold_file)


def _fold_report_slices(lconf, fold):
    """
    Report slices for a given fold
    """
    print('Scoring fold {}...'.format(fold),
          file=sys.stderr)
    dkeys = [econf.key for econf in DETAILED_EVALUATIONS]
    for econf in EVALUATIONS:
        p_path = decode_output_path(lconf, econf, fold)
        enable_details = econf.key in dkeys
        stripped_decoder_key = econf.decoder.key[len(econf.settings.key) + 1:]
        config = (econf.learner.key,
                  stripped_decoder_key,
                  econf.settings.key)
        yield Slice(fold, config,
                    load_predictions(p_path),
                    enable_details)


def _mk_report(lconf, dconf, slices, fold):
    """helper for report generation

    :type fold: int or None
    """
    rpack = full_report(dconf.pack, dconf.folds, slices)
    rpack.dump(report_dir_path(lconf, fold))
    for rconf in LEARNERS:
        if rconf.attach.payload == 'oracle':
            pass
        elif rconf.relate is not None and rconf.relate.payload == 'oracle':
            pass
        else:
            _mk_model_summary(lconf, dconf, rconf, fold)


def _mk_model_summary(lconf, dconf, rconf, fold):
    "generate summary of best model features"
    _top_n = 3

    def _write_discr(discr, intra):
        "write discriminating features to disk"
        if discr is None:
            print(('No discriminating features for {name} {grain} model'
                   '').format(name=rconf.key,
                              grain='sent' if intra else 'doc'),
                  file=sys.stderr)
            return
        output = model_info_path(lconf, rconf, fold, intra)
        with codecs.open(output, 'wb', 'utf-8') as fout:
            print(attelo.report.show_discriminating_features(discr),
                  file=fout)

    labels = dconf.pack.labels
    vocab = load_vocab(vocab_path(lconf))
    # doc level discriminating features
    if True:
        models = attelo_doc_model_paths(lconf, rconf, fold).fmap(load_model)
        discr = attelo.score.discriminating_features(models, labels, vocab,
                                                     _top_n)
        _write_discr(discr, False)

    # sentence-level
    spaths = attelo_sent_model_paths(lconf, rconf, fold)
    if fp.exists(spaths.attach) and fp.exists(spaths.relate):
        models = spaths.fmap(load_model)
        discr = attelo.score.discriminating_features(models, labels, vocab,
                                                     _top_n)
        _write_discr(discr, True)


def _mk_hashfile(parent_dir, lconf, dconf):
    "Hash the features and models files for long term archiving"

    hash_me = [features_path(lconf)]
    for fold in sorted(frozenset(dconf.folds.values())):
        for rconf in LEARNERS:
            models_path = eval_model_path(lconf, rconf, fold, '*')
            hash_me.extend(sorted(glob.glob(models_path + '*')))
    with open(fp.join(parent_dir, 'hashes.txt'), 'w') as stream:
        for path in hash_me:
            fold_basename = fp.basename(fp.dirname(path))
            if fold_basename.startswith('fold-'):
                nice_path = fp.join(fold_basename, fp.basename(path))
            else:
                nice_path = fp.basename(path)
            print('\t'.join([nice_path, md5sum_file(path)]),
                  file=stream)


def _mk_global_report(lconf, dconf):
    "Generate reports for all folds"
    slices = itr.chain.from_iterable(_fold_report_slices(lconf, f)
                                     for f in frozenset(dconf.folds.values()))
    _mk_report(lconf, dconf, slices, None)

    report_dir = report_dir_path(lconf, None)
    final_report_dir = fp.join(lconf.eval_dir,
                               report_dir_basename(lconf))
    mk_graphs(lconf, dconf)
    _mk_hashfile(report_dir, lconf, dconf)
    if fp.exists(final_report_dir):
        shutil.rmtree(final_report_dir)
    shutil.copytree(report_dir, final_report_dir)
    # this can happen if resuming a report; better copy
    # it again
    print('Report saved in ', final_report_dir,
          file=sys.stderr)


def _do_fold(lconf, dconf, fold):
    """
    Run all learner/decoder combos within this fold
    """
    fold_dir = fold_dir_path(lconf, fold)
    print(_fold_banner(lconf, fold), file=sys.stderr)
    if not os.path.exists(fold_dir):
        os.makedirs(fold_dir)

    # learn all models in parallel
    include_intra = any(e.settings.intra is not None
                        for e in EVALUATIONS)
    learner_jobs = concat_i(_delayed_learn(lconf, dconf, rconf, fold,
                                           include_intra)
                            for rconf in LEARNERS)
    _parallel(lconf)(learner_jobs)
    # run all model/decoder joblets in parallel
    decoder_jobs = concat_i(_delayed_decode(lconf, dconf, econf, fold)
                            for econf in EVALUATIONS)
    _parallel(lconf)(decoder_jobs)
    for econf in EVALUATIONS:
        _post_decode(lconf, dconf, econf, fold)
    fold_dir = fold_dir_path(lconf, fold)
    slices = _fold_report_slices(lconf, fold)
    _mk_report(lconf, dconf, slices, fold)


def _mk_combined_models(lconf, dconf):
    """
    Create global for all learners
    """
    include_intra = any(e.settings.intra is not None
                        for e in EVALUATIONS)
    jobs = concat_i(_delayed_learn(lconf, dconf, learner, None, include_intra)
                    for learner in LEARNERS)
    _parallel(lconf)(jobs)


def _is_standalone_or(lconf, stage):
    """
    True if we are in standalone mode (do everything)
    or in a given cluster stage
    """
    return lconf.stage is None or lconf.stage == stage


def _do_corpus(lconf):
    "Run evaluation on a corpus"
    print(_corpus_banner(lconf), file=sys.stderr)

    edus_file = edu_input_path(lconf)
    if not os.path.exists(edus_file):
        _exit_ungathered()

    has_stripped = (lconf.stage in [ClusterStage.end, ClusterStage.start]
                    and fp.exists(features_path(lconf, stripped=True)))
    dpack = load_data_pack(edus_file,
                           pairings_path(lconf),
                           features_path(lconf, stripped=has_stripped),
                           verbose=True)

    if _is_standalone_or(lconf, ClusterStage.start):
        _generate_fold_file(lconf, dpack)

    dconf = DataConfig(pack=dpack,
                       folds=load_fold_dict(lconf.fold_file))

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
    psr.add_argument("--n-jobs", type=int,
                     default=-1,
                     help="number of jobs (-1 for max [DEFAULT], "
                     "2+ for parallel, "
                     "1 for sequential but using parallel infrastructure, "
                     "0 for fully sequential)")
    psr.add_argument("--jumpstart", action='store_true',
                     help="copy any model files over from last evaluation "
                     "(useful if you just want to evaluate recent changes "
                     "to the decoders without losing previous scores)")

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
    _sanity_check_config()
    stage = args_to_stage(args)
    data_dir = latest_tmp()
    if not os.path.exists(data_dir):
        _exit_ungathered()
    eval_dir, scratch_dir = _create_eval_dirs(args, data_dir, args.jumpstart)

    for corpus in TRAINING_CORPORA:
        dataset = os.path.basename(corpus)
        fold_file = os.path.join(eval_dir,
                                 "folds-%s.json" % dataset)

        lconf = LoopConfig(eval_dir=eval_dir,
                           scratch_dir=scratch_dir,
                           folds=args.folds,
                           stage=stage,
                           fold_file=fold_file,
                           n_jobs=args.n_jobs,
                           dataset=dataset)
        _do_corpus(lconf)
