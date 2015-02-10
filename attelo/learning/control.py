"""
Central interface to the learners
"""

from attelo.io import Torpor
from attelo.table import for_attachment, for_labelling
from attelo.util import Team

# pylint: disable=too-few-public-methods

# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


def learn_attach(learners, dpack, verbose=False):
    """
    Train attachment learner
    """
    with Torpor("training attachment model", quiet=not verbose):
        attach_pack = for_attachment(dpack)
        return learners.attach.fit(attach_pack.data,
                                   attach_pack.target)


def learn_relate(learners, dpack, verbose=False):
    """
    Train relation learner
    """
    with Torpor("training relations model", quiet=not verbose):
        relate_pack = for_labelling(dpack.attached_only())
        return learners.relate.fit(relate_pack.data,
                                   relate_pack.target)


def learn(learners, dpack, verbose=False):
    """
    Train learners for each attelo task. Return the resulting
    models

    :type learners: Team(learner)

    :rtype Team(model)
    """
    return Team(attach=learn_attach(learners, dpack, verbose),
                relate=learn_relate(learners, dpack, verbose))
