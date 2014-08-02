# pylint: disable=W0105
"""
Paths and settings used for this experimental harness
In the future we may move this to a proper configuration file.
"""

# Author: Eric Kow
# License: CeCILL-B (French BSD3-like)

import itertools

from attelo.harness.config import\
    LearnerConfig,\
    DecoderConfig,\
    EvaluationConfig

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

PTB_DIR='ptb3'
"""
Where to read the Penn Treebank from (should be dir corresponding to
parsed/mrg/wsj)
"""

LEARNERS = [LearnerConfig.simple("bayes"),
            LearnerConfig.simple("maxent")]
"""Attelo learner algorithms to try
If the second element is None, we use the same learner for attachment
and relations; otherwise we use the first for attachment and the second
for relations
"""

DECODERS = [DecoderConfig.simple(x) for x in
            ["last", "local", "locallyGreedy", "mst"]]
"""Attelo decoders to try in experiment"""


# TODO: make this more elaborate as the configs get more complicated
def _mk_econf_name(learner, decoder):
    """
    generate a short unique name for a learner/decoder combo
    """

    return "%s-%s" % (learner.name, decoder.name)

EVALUATIONS = [EvaluationConfig(name=_mk_econf_name(l, d),
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
