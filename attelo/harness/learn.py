'''
Control over attelo learners as might be needed for a test harness
'''

from __future__ import print_function
from os import path as fp

from joblib import (delayed)
from .util import (makedirs)
from ..io import (Torpor, save_model)
from ..learning import (learn_task)


def learn(dpack, learners, task, output_path, quiet):
    'learn a model and write it to the given output path'
    msg = ("training {task} model {path}"
           "").format(task=task.name,
                      path=fp.basename(output_path))
    with Torpor(msg, sameline=False, quiet=quiet):
        model = learn_task(dpack, learners, task)
        makedirs(fp.dirname(output_path))
        save_model(output_path, model)


def jobs(dpack, learners, tasks, quiet=False):
    """
    Return a list of delayed decoding jobs for a given group
    of learners

    :param tasks: at once the tasks we want to learn
                            for, and the output paths the model
                            for those tasks should go
    :type tasks: dict(Task, filepath)
    """
    return [delayed(learn)(dpack, learners, task, path, quiet)
            for task, path in tasks.items()]
