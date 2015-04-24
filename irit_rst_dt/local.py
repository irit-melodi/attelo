# pylint: disable=W0105
# pylint: disable=star-args
"""
Paths and settings used for this experimental harness
In the future we may move this to a proper configuration file.
"""

# Author: Eric Kow
# License: CeCILL-B (French BSD3-like)

from __future__ import print_function
import itertools as itr

from numpy import inf

from attelo.harness.config import (EvaluationConfig,
                                   LearnerConfig,
                                   Keyed)

from attelo.decoding import (DecodingMode)
from attelo.decoding.baseline import (LocalBaseline)
from attelo.decoding.local import (AsManyDecoder, BestIncomingDecoder)
from attelo.decoding.mst import (MstDecoder, MstRootStrategy)
from attelo.decoding.intra import (IntraInterDecoder, IntraStrategy)
from attelo.learning.perceptron import (Perceptron,
                                        PerceptronArgs,
                                        PassiveAggressive,
                                        StructuredPerceptron,
                                        StructuredPassiveAggressive)
from attelo.learning import (can_predict_proba)
from sklearn.linear_model import (LogisticRegression,
                                  Perceptron as SkPerceptron,
                                  PassiveAggressiveClassifier as
                                  SkPassiveAggressiveClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from .attelo_cfg import (combined_key,
                         Settings,
                         KeyedDecoder,
                         IntraFlag)

# PATHS

LOCAL_TMP = 'TMP'
"""Things we may want to hold on to (eg. for weeks), but could
live with throwing away as needed"""

SNAPSHOTS = 'SNAPSHOTS'
"""Results over time we are making a point of saving"""

# TRAINING_CORPUS = 'tiny'
# TRAINING_CORPUS = 'corpus/RSTtrees-WSJ-main-1.0/TRAINING'
TRAINING_CORPUS = 'corpus/RSTtrees-WSJ-double-1.0'
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

TEST_CORPUS = None
# TEST_CORPUS = 'tiny'
"""Corpora for use in FINAL testing.

You should probably leave this set to None until you've tuned and
tweaked things to the point of being able to write up a paper.
Wouldn't want to overfit to the test corpus, now would we?

(If you leave this set to None, we will perform 10-fold cross
validation on the training data)
"""

TEST_EVALUATION_KEY = None
# TEST_EVALUATION_KEY = 'maxent-AD.L_jnt-mst'
"""Evaluation to use for testing.

Leave this to None until you think it's OK to look at the test data.
The key should be the evaluation key from one of your EVALUATIONS,
eg. 'maxent-C0.9-AD.L_jnt-mst'

(HINT: you can join them together from the report headers)
"""


PTB_DIR = 'ptb3'
"""
Where to read the Penn Treebank from (should be dir corresponding to
parsed/mrg/wsj)
"""

FEATURE_SET = 'dev'  # one of ['dev', 'eyk', 'li2014']
"""
Which feature set to use for feature extraction
"""


def decoder_local(settings):
    "our instantiation of the local baseline decoder"
    use_prob = settings.mode != DecodingMode.post_label
    return LocalBaseline(0.2, use_prob)


def decoder_mst(settings):
    "our instantiation of the local baseline decoder"
    use_prob = settings.mode != DecodingMode.post_label
    return MstDecoder(MstRootStrategy.fake_root,
                      use_prob)

def learner_oracle():
    "return a keyed instance of the oracle (virtual) learner"
    return Keyed('oracle', 'oracle')


def learner_maxent():
    "return a keyed instance of maxent learner"
    return Keyed('maxent', LogisticRegression())


def learner_dectree():
    "return a keyed instance of decision tree learner"
    return Keyed('dectree', DecisionTreeClassifier())


def learner_rndforest():
    "return a keyed instance of random forest learner"
    return Keyed('rndforest', RandomForestClassifier())


LOCAL_PERC_ARGS = PerceptronArgs(iterations=20,
                                 averaging=True,
                                 use_prob=False,
                                 aggressiveness=inf)

LOCAL_PA_ARGS = PerceptronArgs(iterations=20,
                               averaging=True,
                               use_prob=False,
                               aggressiveness=inf)

STRUCT_PERC_ARGS = PerceptronArgs(iterations=50,
                                  averaging=True,
                                  use_prob=False,
                                  aggressiveness=inf)

STRUCT_PA_ARGS = PerceptronArgs(iterations=50,
                                averaging=True,
                                use_prob=False,
                                aggressiveness=inf)

_LOCAL_LEARNERS = [
    LearnerConfig(attach=learner_oracle(),
                  relate=learner_oracle()),
    LearnerConfig(attach=learner_maxent(),
                  relate=learner_maxent()),
    LearnerConfig(attach=learner_maxent(),
                  relate=learner_oracle()),
    LearnerConfig(attach=learner_rndforest(),
                  relate=learner_rndforest()),
#    LearnerConfig(attach=Keyed('sk-perceptron',
#                               SkPerceptron(n_iter=20)),
#                  relate=learner_maxent()),
#    LearnerConfig(attach=Keyed('sk-pasagg',
#                               SkPassiveAggressiveClassifier(n_iter=20)),
#                  relate=learner_maxent()),
#    LearnerConfig(attach=Keyed('dp-perc',
#                               Perceptron(d, LOCAL_PERC_ARGS)),
#                  relate=learner_maxent()),
#    LearnerConfig(attach=Keyed('dp-pa',
#                               PassiveAggressive(d, LOCAL_PA_ARGS)),
#                  relate=learner_maxent()),
]
"""Straightforward attelo learner algorithms to try

It's up to you to choose values for the key field that can distinguish
between different configurations of your learners.

"""

_STRUCTURED_LEARNERS = [
#    lambda d: LearnerConfig(attach=Keyed('dp-struct-perc',
#                                         StructuredPerceptron(d, STRUCT_PERC_ARGS)),
#                            relate=learner_maxent()),
#    lambda d: LearnerConfig(attach=Keyed('dp-struct-pa',
#                                         StructuredPassiveAggressive(d, STRUCT_PA_ARGS)),
#                            relate=learner_maxent()),
]

"""Attelo learners that take decoders as arguments.
We assume that they cannot be used relation modelling
"""


_CORE_DECODERS = [
    Keyed('local', decoder_local),
    Keyed('mst', decoder_mst),
#    Keyed('asmany', lambda _: AsManyDecoder()),
#    Keyed('bestin', lambda _: BestIncomingDecoder()),
]

"""Attelo decoders to try in experiment

Don't forget that you can parameterise the decoders ::

    Keyed('astar-3-best' decoder_astar(nbest=3))
"""


_SETTINGS = [
    Settings(key='AD.L_jnt_intra_soft',
             mode=DecodingMode.joint,
             intra=IntraFlag(strategy=IntraStrategy.soft,
                             intra_oracle=False,
                             inter_oracle=False)),
    Settings(key='AD.L_jnt_intra_heads',
             mode=DecodingMode.joint,
             intra=IntraFlag(strategy=IntraStrategy.heads,
                             intra_oracle=False,
                             inter_oracle=False)),
    Settings(key='AD.L_jnt_intra_only',
             mode=DecodingMode.joint,
             intra=IntraFlag(strategy=IntraStrategy.only,
                             intra_oracle=False,
                             inter_oracle=False)),
    Settings(key='AD.L_jnt',
             mode=DecodingMode.joint,
             intra=None),
    Settings(key='AD.L_pst',
             mode=DecodingMode.post_label,
             intra=None),
    ]
"""Variants on global settings that would generally apply
over all decoder combos.

    Variant(key="post-label",
            name=None,
            flags=["--post-label"])

The name field is ignored here.

Note that not all global settings may be applicable to
all decoders.  For example, some learners may only
supoort '--post-label' decoding.

You may need to write some fancy logic when building the
EVALUATIONS list below in order to exclude these
possibilities
"""


def _is_junk(klearner, kdecoder):
    """
    Any configuration for which this function returns True
    will be silently discarded
    """
    # intrasential head to head mode only works with mst for now
    intra_flag = kdecoder.settings.intra
    if kdecoder.key != 'mst':
        if (intra_flag is not None and
                intra_flag.strategy == IntraStrategy.heads):
            return True

    # no need for intra/inter oracle mode if the learner already
    # is an oracle
    if klearner.key == 'oracle' and intra_flag is not None:
        if intra_flag.intra_oracle or intra_flag.inter_oracle:
            return True

    # skip any config which tries to use a non-prob learner with
    if not can_predict_proba(klearner.attach.payload):
        if kdecoder.settings.mode != DecodingMode.post_label:
            return True

    return False


def _mk_keyed_decoder(kdecoder, settings):
    """construct a decoder from the settings

    :type k_decoder: Keyed(Settings -> Decoder)

    :rtype: KeyedDecoder
    """
    decoder_key = combined_key([settings, kdecoder])
    decoder = kdecoder.payload(settings)
    if settings.intra:
        decoder = IntraInterDecoder(decoder, settings.intra.strategy)
    return KeyedDecoder(key=decoder_key,
                        payload=decoder,
                        settings=settings)


def _mk_evaluations():
    """
    Some things we're trying to capture here:

    * some (fancy) learners are parameterised by decoders

    Suppose we have decoders (local, mst, astar) and the learners
    (maxent, struct-perceptron), the idea is that we would want
    to be able to generate the models:

        maxent (no parameterisation with decoders)
        struct-perceptron-mst
        struct-perceptron-astar

    * in addition to decoders, there are variants on global
      decoder settings that we want to expand out; however,
      we do not want to expand this for purposes of model
      learning

    * if a learner is parameterised by a decoder, it should
      only be tested by the decoder it is parameterised
      against (along with variants on its global settings)

        - struct-perceptron-mst with the mst decoder
        - struct-perceptron-astar with the astar decoder

    * ideally (not mission-critical) we want to report all the
      struct-perceptron-* learners as struct-percepntron; but
      it's easy to accidentally do the wrong thing, so let's not
      bother, eh?

    This would be so much easier with static typing

    :rtype [(Keyed(learner), KeyedDecoder)]
    """

    kdecoders = [_mk_keyed_decoder(d, s)
                 for d, s in itr.product(_CORE_DECODERS, _SETTINGS)]
    kdecoders = [k for k in kdecoders if k is not None]

    # all learner/decoder pairs
    pairs = []
    pairs.extend(itr.product(_LOCAL_LEARNERS, kdecoders))
    for klearner in _STRUCTURED_LEARNERS:
        pairs.extend((klearner(d.payload), d) for d in kdecoders)

    # boxing this up a little bit more conveniently
    return [EvaluationConfig(key=combined_key([klearner, kdecoder]),
                             settings=kdecoder.settings,
                             learner=klearner,
                             decoder=kdecoder)
            for klearner, kdecoder in pairs
            if not _is_junk(klearner, kdecoder)]


EVALUATIONS = _mk_evaluations()
"""Learners and decoders that are associated with each other.
The idea her is that if multiple decoders have a learner in
common, we will avoid rebuilding the model associated with
that learner.  For the most part we just want the cartesian
product, but some more sophisticated learners depend on the
their decoder, and cannot be shared
"""


GRAPH_DOCS = [
    'wsj_1184.out',
    'wsj_1120.out',
    ]
"""Just the documents that you want to graph.
Set to None to graph everything
"""

DETAILED_EVALUATIONS = [e for e in EVALUATIONS if
                        'maxent' in e.learner.key and
                        ('mst' in e.decoder.key or 'astar' in e.decoder.key)
                        and 'jnt' in e.settings.key
                        and 'orc' not in e.settings.key]
"""
Any evalutions that we'd like full reports and graphs for.
You could just set this to EVALUATIONS, but this sort of
thing (mostly the graphs) takes time and space to build

HINT: set to empty list for no graphs whatsoever
"""


def print_evaluations():
    """
    Print out the name of each evaluation in our config
    """
    for econf in EVALUATIONS:
        print(econf)
        print()
    print("\n".join(econf.key for econf in EVALUATIONS))

if __name__ == '__main__':
    print_evaluations()
