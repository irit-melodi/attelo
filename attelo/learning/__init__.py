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
