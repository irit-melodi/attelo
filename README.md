These are experiments being conducted around 2014-07 by the
Melodi team in IRIT on the RST Discourse Treebank corpus.

## Prerequisites

1. Python 2.7. Python 3 might also work.
2. pip 1.5 or greater
3. git (to keep up with educe/attelo changes)
4. [optional] Anacoda (`conda --version` should say 2.2.6 or higher)
5. a copy of the RST Discourse Treebank

## Sandboxing

If you are attempting to use the development version of this code
(ie. from SVN), I highly recommend using a sandbox environment,
particularly conda (Anaconda on Mac, Miniconda on Linux)

## Installation (development mode)

1. Create your virtual environment

   ```
   conda create -n irit-rst-dt scipy
   ```

2. Activate the virtual environment (note that you'll need to do
   this whenever you want to run the harness)

   ```
   source activate irit-rst-dt
   ```

3. Install this package and its dependencies

   ```
   pip install -r requirements.txt
   pip install -e .
   ```

   (N.B. By rights the second step `pip install -e` is redundant with
   the first; however, for some mysterious reason, the irit-rst-dt
   script is sometimes not installed if you only run the first)

3. Link your copy of the RST DT corpus in, along with the
   Penn Treebank, for example:

   ```
   ln -s $HOME/CORPORA/rst_discourse_treebank/data corpus
   ln -s $HOME/CORPORA/PTBIII/parsed/mrg/wsj ptb3
   ```

## Cluster

Do you have access to a fancy compute cluster? You can use it to speed
things up (mostly by taking advantage of parallelism).  If it's using
SLURM, check cluster/README.md

## Preflight checklist

* Have you linked the corpora in? (see Installation)

## Usage

Running the pieces of infrastructure here should consist of running
`irit-rst-dt <subcommand>`

### Configuration

Have a look at the `irit_rst_dt.local` module. You may want to modify
which subcorpus we run this on. I would suggest making a sample out of
20 files or so and working with that until you are familiar with the
harness first. You could likewise also consider reducing the learners
and decoders you want to experiment with initially.

Note that the current configuration assumes that learners and decoders
are independent of each other. If you have something like the structured
perceptron learner which depends on its decoder, talk to Eric. Likewise,
we don't yet offer a way to change the configuration settings for the
learners/decoders. Talk to Eric about improving the harness.

### Basics

Using the harness consists of two steps, gathering the features, and
running the n-fold cross validation loop

    irit-rst-dt gather
    irit-rst-dt evaluate

If you stop an evaluation (control-C) in progress, you can resume it
by running

    irit-rst-dt evaluate --resume

The harness will try to detect what work it has already done and pick
up where it left off.

### Scores

You can get a sense of how things are going by inspecting the various
intermediary results

1. count files: At each learner/decoder combination we will count
   the number of correct attachments and relation labels (and save
   these into intermediary count) (for a given fold N, see
   `TMP/latest/scratch-current/fold-N/count-*.csv`)

2. fold summaries: At the end of each fold, we will summarise all of
   the counts into a simple Precision/Recall/F1 report for attachment
   and labelling. (for a given fold N, see
   `TMP/latest/scratch-current/fold-N/scores*.txt`)

3. full summary: If we make it through the entire experiment, we will
   produce a cross-validation summary combining the counts from all
   folds (`TMP/latest/eval-current/scores*.txt`)

### Cleanup

The harness produces a lot of output, and can take up potentially a lot
of disk space in the process.  If you have saved results you want to
keep, you can run the command

    irit-rst-dt clean

This will delete *all* scratch directories, along with any evaluation
directories that look incomplete (no scores).

### Output files

There are two main directories for output.

* The SNAPSHOTS directory is meant for intermediary results that you want
to save. You have to copy files into here manually (more on that later).
Because this directory can take up space, it does not feel quite right
to dump it on the public GitHub repo. We'll need to think about where to
store our snapshots later (possibly some IRIT-local SVN?)

* The TMP directory is where the test harness does all its work.  Each
`TMP/<timestamp>` directory corresponds to a set of feature files
generated by `irit-rst-dt gather`.  For convenience, the harness will
maintain a `TMP/latest` symlink pointing to one of these directories.

Within the each feature directory, we can have a number of evaluation
and scratch directories. This layout is motivated by us wanting to
suppport ongoing changes to our learning/decoding algorithms
independently of feature collection. So the thinking is that we may
have multiple evaluations for any given set of features. Like the
feature directories, these are named by timestamp (with
`eval-current` and `scratch-current` symlinks for convenience).

* scratch directories: these are considered relatively ephemeral
  (hence them being deleted by `irit-rst-dt clean`). They contain
  all the models and counts saved by harness during evaluation.

* eval directories: these contain things we would consider more
  essential for reproducing an evaluation. They contain the
  feature files (hardlinked from the parent dir) along with the
  fold listing and the cross-fold validation scores. If you hit
  any interesting milestones in development, it may be good to
  manually copy the eval directory to SNAPSHOTS, maybe with a
  small README explaining what it is, or at least a vaguely
  memorable name. This directory should be fairly self-contained.
