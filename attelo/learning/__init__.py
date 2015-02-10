'''
attelo learners: this is mostly just bog standard learners provided
by scikit-learn, with some custom experimental ones thrown in for good
measure (they should be roughly compatible though)
'''

from collections import namedtuple
import copy

from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from .control import (learn,
                      learn_attach,
                      learn_relate)
from .perceptron import (PerceptronArgs,
                         Perceptron,
                         PassiveAggressive,
                         StructuredPerceptron,
                         StructuredPassiveAggressive)

# pylint: disable=too-few-public-methods


class LearnerArgs(namedtuple("LearnerArgs",
                             ["decoder",
                              "perc_args"])):
    '''
    Parameters used to instantiate attelo learners.
    Not all parameters are used by all learners
    '''


_LEARNERS =\
    {"bayes": lambda _: MultinomialNB(),
     "maxent": lambda _: LogisticRegression(),
     "svm": lambda _: SVC(),
     "majority": lambda _: DummyClassifier(strategy="most_frequent")}

ATTACH_LEARNERS = copy.copy(_LEARNERS)
'''
learners that can be used for the attachment task

dictionary from learner names (recognised by the attelo command
line interface) to learner wrappers

The wrappers must accept a :py:class:LearnerArgs: tuple,
the idea being that it would pick out any parameters relevant to it
and ignore the rest
'''
ATTACH_LEARNERS["constant"] = lambda _: DummyClassifier(strategy="constant",
                                                        constant=1)


# TODO (Pascal)
#
# Insert perceptrons into ATTACH_LEARNERS dict

RELATE_LEARNERS = copy.copy(_LEARNERS)
'''
learners that can be used for the relation labelling task

(see `ATTACH_LEARNERS`)
'''
