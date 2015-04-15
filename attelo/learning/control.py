"""
Central interface to the learners
"""

from enum import Enum

from attelo.table import (for_attachment,
                          for_labelling)
from attelo.util import Team

# pylint: disable=too-few-public-methods

# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


class Task(Enum):
    '''
    Learning tasks for a model.
    '''
    attach = 1
    relate = 2


def learn_task(mpack, learners, task):
    """
    Train attachment learner
    """
    if task == Task.attach:
        mpack = {k: for_attachment(v)
                 for k, v in mpack.items()}
        dpacks = mpack.values()
        targets = [d.target for d in dpacks]
        return learners.attach.fit(dpacks, targets)
    elif task == Task.relate:
        mpack = {k: for_labelling(v.attached_only())
                 for k, v in mpack.items()}
        dpacks = mpack.values()
        targets = [d.target for d in dpacks]
        return learners.relate.fit(dpacks, targets)
    else:
        raise ValueError('Unknown learning task: {}'.format(task))


def learn(dpack, learners):
    """
    Train learners for each attelo task. Return the resulting
    models

    :type learners: Team(learner)

    :rtype Team(model)
    """
    return Team(attach=learn_task(dpack, learners, Task.attach),
                relate=learn_task(dpack, learners, Task.relate))


def can_fit_structured(learner):
    """
    True if a learner can work with structured input, ie. taking
    in lists of vector-lists (actually datapacks) and lists of
    target-lists instead of a flat list of vectors and targets.
    """
    func = getattr(learner, "fit_structured", None)
    return callable(func)
