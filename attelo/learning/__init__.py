'''
attelo learners: this is mostly just bog standard learners provided
by scikit-learn, with some custom experimental ones thrown in for good
measure (they should be roughly compatible though)

Learners
--------
Following scikit conventions, a learner is an object that, once
fitted to some training data, becomes a classifier (which can be
used to make predictions).

For now we support two learner styles.

* local learners implement a `fit(X, y)` function which take a
  feature and data matrix as input
* structured learners implement a `fit_structured(Xs, ys)`
  basically feature matrix and a target vectors, one per
  document

Attachment classifiers
----------------------
For purposes of predicting attachment, a classifier must implement
either

* `predict_proba(X)`: given a feature matrix return a probability
   matrix, each row being the vector of probabilities that a pair
   is unattached or attached (in pracitce we're just interested in
   the latter)
* `decision_function(X)`: given a feature matrix, return a vector
   of attachment scores

Note that these come directly out of scikit.

Label classifiers
-----------------
For purposes of predicting relation labels, a classifier must
implement the following functions

* `predict_proba(X)`: given a feature matrix, return a probability
   matrix, each row being a distribution of probabilities over the
   set of possible labels

* `predict(X)`: given a feature matrix, return a vector containing
  the best label for each row
'''

from collections import namedtuple
import copy

from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import (LogisticRegression,
                                  Perceptron as SkPerceptron,
                                  PassiveAggressiveClassifier as
                                  SkPassiveAggressiveClassifier)
from sklearn.svm import SVC

from .control import (Task,
                      learn,
                      learn_task)
from .local import (SklearnAttachClassifier,
                    SklearnLabelClassifier)
from .oracle import (AttachOracle,
                     LabelOracle)


from .perceptron import (PerceptronArgs,
                         Perceptron,
                         PassiveAggressive,
                         StructuredPerceptron,
                         StructuredPassiveAggressive)
# pylint: disable=wildcard-import
from .interface import *
# pylint: enable=wildcard-import

# pylint: disable=too-few-public-methods


class LearnerArgs(namedtuple("LearnerArgs",
                             ["decoder",
                              "perc_args"])):
    '''
    Parameters used to instantiate attelo learners.
    Not all parameters are used by all learners
    '''


ATTACH_LEARNERS =\
    {"oracle": lambda _: AttachOracle(),
     "bayes": lambda _: SklearnAttachClassifier(MultinomialNB()),
     "maxent": lambda _: SklearnAttachClassifier(LogisticRegression()),
     "svm": lambda _: SklearnAttachClassifier(SVC()),
     "majority": lambda _: SklearnAttachClassifier(DummyClassifier(strategy="most_frequent")),
     "always": lambda _: SklearnAttachClassifier(DummyClassifier(strategy="constant",
                                                                 constant=1)),
     "never": lambda _: SklearnAttachClassifier(DummyClassifier(strategy="constant",
                                                                constant=-1)),
     "sk-perceptron": lambda _: SklearnAttachClassifier(SkPerceptron()),
     "sk-pasagg": lambda _: SklearnAttachClassifier(SkPassiveAggressiveClassifier())}


'''
learners that can be used for the attachment task

dictionary from learner names (recognised by the attelo command
line interface) to learner wrappers

The wrappers must accept a :py:class:LearnerArgs: tuple,
the idea being that it would pick out any parameters relevant to it
and ignore the rest
'''
# # local reimplemented learners
# ATTACH_LEARNERS["perc"] = lambda c: Perceptron( c.perc_args )
# ATTACH_LEARNERS["pa"] = lambda c: PassiveAggressive( c.perc_args )

# # structured learners
# ATTACH_LEARNERS["perc-struct"] = lambda c: StructuredPerceptron( c.perc_args )
# ATTACH_LEARNERS["pa-struct"] = lambda c: StructuredPassiveAggressive( c.perc_args )

RELATE_LEARNERS =\
    {"oracle": lambda _: LabelOracle(),
     "bayes": lambda _: SklearnLabelClassifier(MultinomialNB()),
     "maxent": lambda _: SklearnLabelClassifier(LogisticRegression()),
     "svm": lambda _: SklearnLabelClassifier(SVC()),
     "majority": lambda _: SklearnLabelClassifier(DummyClassifier(strategy="most_frequent"))}

'''
learners that can be used for the relation labelling task

(see `ATTACH_LEARNERS`)
'''
