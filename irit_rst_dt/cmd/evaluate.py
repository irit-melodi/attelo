# Author: Eric Kow
# License: CeCILL-B (French BSD3-like)

"""
run an experiment
"""

from __future__ import print_function
import os
import subprocess
import sys

from ..config import\
    TRAINING_CORPORA, LEARNERS, DECODERS, ATTELO_CONFIG_FILE
from ..util import\
    latest_tmp, timestamp

NAME = 'evaluate'


def config_argparser(parser):
    """
    Subcommand flags.

    You should create and pass in the subparser to which the flags
    are to be added.
    """
    parser.set_defaults(func=main)


def _banner(dataset, decoder, learner_a, learner_r):
    """
    Which combo of eval parameters are we running now?
    """
    learner_str = learner_a + (":" + learner_r if learner_r else "")
    return "\n".join(["==========" * 6,
                      " ".join(["corpus:", dataset + ",",
                                "learner(s):", learner_str + ",",
                                "decoder:", decoder]),
                      "==========" * 6])


def main(_):
    """
    Subcommand main.

    You shouldn't need to call this yourself if you're using
    `config_argparser`
    """
    data_dir = latest_tmp()
    eval_dir = os.path.join(data_dir, "eval-" + timestamp())
    os.makedirs(eval_dir)
    with open(os.path.join(eval_dir, "versions.txt"), "w") as stream:
        subprocess.check_call(["pip", "freeze"], stdout=stream)

    if not os.path.exists(data_dir):
        sys.exit("""No data to run experiments on.
Please run `irit-rst-dt gather`""")
    for corpus in TRAINING_CORPORA:
        dataset = os.path.basename(corpus)
        scores_file = os.path.join(eval_dir, "scores-%s" % dataset)
        for learner_a, learner_r in LEARNERS:
            for decoder in DECODERS:
                mk_csv_path = lambda x:\
                    os.path.join(data_dir, "%s.%s.csv" % (dataset, x))
                print(_banner(dataset, decoder, learner_a, learner_r),
                      file=sys.stderr)
                with open(scores_file, "a") as scores:
                    cmd = ["attelo",
                           "evaluate",
                           "-C", ATTELO_CONFIG_FILE,
                           "-l", learner_a,
                           "-d", decoder]
                    if learner_r:
                        cmd += ["--relation-learner", learner_r]
                    cmd += [mk_csv_path("edu-pairs"),
                            mk_csv_path("relations")]
                    subprocess.call(cmd, stdout=scores)
