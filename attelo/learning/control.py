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


def learn(learners, dpack, verbose=False):
    """
    Train learners for each attelo task. Return the resulting
    models

    :type learners: Team(learner)

    :rtype Team(model)
    """
    with Torpor("training attachment model", quiet=not verbose):
        attach_pack = for_attachment(dpack)
        attach_model = learners.attach.fit(attach_pack.data,
                                           attach_pack.target)

    with Torpor("training relations model", quiet=not verbose):
        relate_pack = for_labelling(dpack.attached_only())
        relate_model = learners.relate.fit(relate_pack.data,
                                           relate_pack.target)

    print(relate_model.classes_)
    return Team(attach=attach_model,
                relate=relate_model)
