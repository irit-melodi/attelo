# pylint: disable=W0105
"""
Paths and settings used for this experimental harness
In the future we may move this to a proper configuration file.
"""

# Author: Eric Kow
# License: CeCILL-B (French BSD3-like)

import itertools

from attelo.harness.config import (EvaluationConfig,
                                   LearnerConfig,
                                   Variant,
                                   Team)

LOCAL_TMP = 'TMP'
"""Things we may want to hold on to (eg. for weeks), but could
live with throwing away as needed"""

SNAPSHOTS = 'SNAPSHOTS'
"""Results over time we are making a point of saving"""

TRAINING_CORPORA = ['corpus/RSTtrees-WSJ-double-1.0']
"""Corpora for use in building/training models and running our
incremental experiments. Later on we should consider using the
held-out test data for something, but let's make a point of
holding it out for now.

Note that by convention, corpora are identified by their basename.
Something like `corpus/RSTtrees-WSJ-main-1.0/TRAINING` would result
in a corpus named "TRAINING". This could be awkward if the basename
is used in more than one corpus, but we can revisit this scheme as
needed.
"""

PTB_DIR = 'ptb3'
"""
Where to read the Penn Treebank from (should be dir corresponding to
parsed/mrg/wsj)
"""

LEARNERS = [LearnerConfig(attach=Variant(key="bayes", name="bayes", flags=[]),
                          relate=None),
            LearnerConfig(attach=Variant(key="maxent", name="maxent", flags=[]),
                          relate=None)]
"""Attelo learner algorithms to try.  In the general case, you can
leave the relate learner as `None` (in which case we just use the.

It's up to you to choose values for the key field that can distinguish
between different configurations of your learners.  For example,
you might have something like

::

    Variant(key="perceptron-very-frob",
            name="perceptron",
            flags=["--frobnication", "0.9"]),

    Variant(key="perceptron-not-so-frob",
            name="perceptron",
            flags=["--frobnication", "0.4"])

"""

DECODERS = [Variant(key=x, name=x, flags=[]) for x in
            ["last", "local", "locallyGreedy", "mst", "astar"]]
"""Attelo decoders to try in experiment

Don't forget that you can parameterise the decoders ::

    Variant(key="astar-3-best",
            name="astar",
            flags=["--nbest", "3"])
"""


EVALUATIONS = [EvaluationConfig(key=EvaluationConfig.simple_key(l, d),
                                learner=l,
                                decoder=d) for l, d in
               itertools.product(LEARNERS, DECODERS)]
"""Learners and decoders that are associated with each other.
The idea her is that if multiple decoders have a learner in
common, we will avoid rebuilding the model associated with
that learner.  For the most part we just want the cartesian
product, but some more sophisticated learners depend on the
their decoder, and cannot be shared
"""


ATTELO_CONFIG_FILE = "attelo.config"
"""Attelo feature configuration"""
