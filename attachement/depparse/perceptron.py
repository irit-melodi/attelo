# Copyright (c) 2009 Leif Johnson <leif@leifjohnson.net>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

'''A basic one-vs-all multiclass Voted Perceptron algorithm.'''

import logging
import collections

def _dot(fws, fs):
    return sum(fws.get(f, 0) for f in fs)

try:
    import _sparse
    dot = _sparse.dot
except ImportError:
    dot = _dot


class VotedPerceptron(object):
    '''The Voted Perceptron is a fast wide-margin classifier.

    This implementation is completely sparse ; that is, both labels and weights
    for those labels are maintained using dictionaries. All dot products are
    calculated by summing the nonzero feature weights for a set of features.
    New labels are introduced by calling train() ; such labels will start out
    with an empty set of feature weights.
    '''

    def __init__(self, feature_beam=5000):
        '''Create a new Voted Perceptron.

        feature_beam: The maximum number of weights to retain in each class.
        '''
        self._acc_weights = {}
        self._acc_iterations = {}
        self._cur_weights = {}
        self._cur_iterations = {}
        self._feature_beam = feature_beam

    def train(self, features, correct):
        '''Present a labeled feature set to the perceptron for learning.

        features: A set of features for classification.
        correct: The correct class for this feature set.
        '''
        if correct not in self._acc_weights:
            self._acc_weights[correct] = collections.defaultdict(float)
            self._acc_iterations[correct] = 0.0
            self._cur_weights[correct] = collections.defaultdict(float)
            self._cur_iterations[correct] = 0.0

        self._acc_iterations[correct] += 1
        predicted, score = self._max_label_score(features, self._cur_weights)
        if predicted == correct:
            self._cur_iterations[correct] += 1
            return

        self._learn(predicted, features, -1.0)
        self._learn(correct, features, 1.0)

    def score(self, features):
        '''Get a score for a feature set.

        features: A set of features for a scoring decision.

        Return a (label, score) pair.
        '''
        return self._max_label_score(features, self._acc_weights, True)

    def labels(self):
        return tuple(self._acc_weights)

    def top_features(self, num_features):
        '''Iterate over the top features for all classes.

        num_features: The number of features to return for each class.
        '''
        for label, fws in self._acc_weights.iteritems():
            ordered = sorted(fws.iteritems(), key=lambda x: -abs(x[1]))
            yield label, len(ordered), [(f, w / self._acc_iterations[label])
                                        for f, w in ordered[:num_features]]

    def _max_label_score(self, features, weights, divide=False):
        '''Get the maximal class and sum of the weights for a feature vector.

        features: A set of features.
        weights: A map from features to weights.
        divide: If True, divide by the iterations of each label.

        Returns a (label, score) pair where the score is the greatest out of all
        scores for all labels.
        '''
        max_label = None
        max_score = -1e100
        for label, fws in weights.iteritems():
            score = dot(fws, features)
            if divide:
                score /= self._acc_iterations[label]
            if score > max_score:
                max_label = label
                max_score = score
        return max_label, max_score

    def _prune(self, weights):
        '''Retain only the top feature_beam feature weights.'''
        if len(weights) < 1.3 * self._feature_beam:
            return
        keys = sorted(weights, key=lambda k: -abs(weights[k]))
        for k in keys[self._feature_beam:]:
            del weights[k]

    def __iadd__(self, other):
        '''Merge the weights from another classifier into this one.'''
        other.finalize()
        for label, source in other._acc_weights.iteritems():
            self._acc_iterations.setdefault(label, 0.0)
            target = self._acc_weights.setdefault(
                label, collections.defaultdict(float))
            for f, w in source.iteritems():
                target[f] += w
            self._prune(target)
            self._acc_iterations[label] += other._acc_iterations[label]
        return self

    def _learn(self, label, features, delta):
        self._accumulate(label)
        target = self._cur_weights[label]
        for f in features:
            target[f] += delta
        self._prune(target)

    def _accumulate(self, label):
        '''Merge the local weights into the global weights for a label.'''
        s = self._cur_iterations[label]
        if s > 0:
            target = self._acc_weights[label]
            for f, w in self._cur_weights[label].iteritems():
                target[f] += s * w
            self._prune(target)
            self._cur_iterations[label] = 0

    def finalize(self):
        '''Accumulate local weights into global weights.'''
        for label in self._cur_weights:
            self._accumulate(label)
