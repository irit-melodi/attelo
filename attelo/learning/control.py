"""
Central interface to the learners
"""

from enum import Enum

from attelo.table import for_attachment, for_labelling
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


def learn_task(dpack, learners, task):
    """
    Train attachment learner
    """
    if task == Task.attach:
        dpack = for_attachment(dpack)
        learner = learners.attach
    elif task == Task.relate:
        dpack = for_labelling(dpack.attached_only())
        learner = learners.relate
    else:
        raise ValueError('Unknown learning task: {}'.format(task))

    return learner.fit(dpack.data, dpack.target)


def learn(dpack, learners):
    """
    Train learners for each attelo task. Return the resulting
    models

    :type learners: Team(learner)

    :rtype Team(model)
    """
    return Team(attach=learn_task(dpack, learners, Task.attach),
                relate=learn_task(dpack, learners, Task.relate))


def can_predict_proba(model):
    """
    True if a model is capable of returning a probability for
    a given instance. The alternative would be for it to
    implement `decision_function` which associates
    """
    if model == 'oracle':
        return True
    else:
        func = getattr(model, "predict_proba", None)
        return callable(func)
