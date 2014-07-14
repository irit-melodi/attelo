# pylint: disable=W0105
"""
Paths and settings used for this experimental harness
In the future we may move this to a proper configuration file.
"""

# Author: Eric Kow
# License: CeCILL-B (French BSD3-like)

LOCAL_TMP = 'TMP'
"""Things we may want to hold on to (eg. for weeks), but could
live with throwing away as needed"""

SNAPSHOTS = 'SNAPSHOTS'
"""Results over time we are making a point of saving"""

TRAINING_CORPORA = ['corpus/RSTtrees-WSJ-main-1.0/TRAINING']
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

LEARNERS = [("bayes", None),
            ("maxent", None)]
"""Attelo learner algorithms to try
If the second element is None, we use the same learner for attachment
and relations; otherwise we use the first for attachment and the second
for relations
"""

DECODERS = ["last", "local", "locallyGreedy", "mst"]
"""Attelo decoders to try in experiment"""

ATTELO_CONFIG_FILE = "attelo.config"
"""Attelo feature configuration"""
