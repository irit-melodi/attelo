# Copyright (c) 2010 Leif Johnson <leif@leifjohnson.net>
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

'''Base classes for dependency parsers.'''

import sys
import logging
import collections

import parallel
import sentence


class Parser(object):
    '''An abstract base class for parallelized dependency parsers.

    To implement this class, provide function bodies for the learn and parse
    methods. These methods are used to learn and parse an individual sentence,
    respectively, and are called in turn from wrapper methods that manage the
    training and testing processes across distributed workers.
    '''

    def __init__(self, ruleset, build_classifier):
        '''Initialize the parser with a rule set and a beam width.

        ruleset: Some sort of feature extraction rules. The contract between
          this object and the parser is completely flexible ; the only important
          thing here is that the implementaions of parse() and train() must obey
          the ruleset contract.
        build_classifier: A callable that takes one argument---the number of
          classes---and returns a multiclass classifier. Again, the contract
          between the classifier and the parser is completely open.
        '''
        self._ruleset = ruleset
        self._build_classifier = build_classifier
        if not callable(build_classifier):
            logging.critical('Error ! build_classifier must be callable.')
            sys.exit(1)
        self._action_classifier = build_classifier()
        self._label_classifier = build_classifier()

    def __getstate__(self):
        return self._ruleset, self._action_classifier, self._label_classifier

    def __setstate__(self, state):
        self._ruleset, self._action_classifier, self._label_classifier = state

    def labels(self):
        '''Return a tuple with the labels that our parser knows about.'''
        return self._label_classifier.labels()

    def log_top_features(self, num_features):
        '''Print out information about the top features for all labels.

        num_features: The number of features to log per label.
        '''
        def log_label_weights(metalabel, label, count, fws):
            logging.error('top %d of %d features for %s %s:%s',
                          len(fws), count, metalabel, label,
                          ''.join('\n\t%7.4f\t%s' % (w, f) for f, w in fws))
        for args in self._action_classifier.top_features(num_features):
            log_label_weights('action', *args)
        for args in self._label_classifier.top_features(num_features):
            log_label_weights('label', *args)

    def learn(self, sent, action_classifier, label_classifier, **kwargs):
        '''Learn from a labeled training Sentence.

        sent: A labeled training Sentence.
        action_classifier: The action classifier to train.
        label_classifier: The label classifier to train.
        '''
        raise NotImplementedError

    def train(self, sents, concurrency=1, **kwargs):
        '''Train on a set of sentences.

        sents: A set of sentences to train on.
        concurrency: Use this many parallel processes for training.

        The remainder of the keyword arguments are passed to the _training_args
        method to extract arguments for running the parallel training processes.
        '''
        if concurrency > 1:
            def build_classifiers(index):
                return (self._build_classifier(), self._build_classifier())

            logging.info('training: %d concurrent processes', concurrency)
            result_q = parallel.launch(concurrency,
                                       sents,
                                       parallel._train,
                                       self,
                                       mkargs=build_classifiers,
                                       **kwargs)
            for _ in xrange(concurrency):
                ac, lc = result_q.get()
                self._action_classifier += ac
                self._label_classifier += lc

        else:
            logging.info('training: serially')
            for sent in sents:
                logging.debug('training on [%s]', sent)
                self.learn(sent,
                           self._action_classifier,
                           self._label_classifier,
                           **kwargs)

        self._action_classifier.finalize()
        self._label_classifier.finalize()

    def parse(self, sent, **kwargs):
        '''Parse an unlabeled test Sentence.

        sent: An unlabeled test Sentence.

        Return the Estimate that the parser constructs for the Sentence.
        '''
        raise NotImplementedError

    def test(self, sents, concurrency=1, **kwargs):
        '''Test in parallel on a set of unseen-during-training Sentences.

        sents: A set of Sentences to evaluate for testing.
        concurrency: Use this many parallel processes for testing.
        '''
        evaluator = Evaluator()

        if concurrency > 1:
            logging.info('testing: %d concurrent processes', concurrency)
            result_q = parallel.launch(
                concurrency, sents, parallel._test, self, **kwargs)
            for _ in sents:
                evaluator.measure(*result_q.get())

        else:
            logging.info('testing: serially')
            for sent in sents:
                evaluator.measure(est=self.parse(sent, **kwargs), sent=sent)

        return evaluator


class Estimate(sentence.Sequence):
    '''A base class for holding estimates of sequence values.'''

    def __init__(self, sent):
        '''Initialize this estimate to point at a Sentence.'''
        # we have our own estimates for head() and label() ...
        super(Estimate, self).__init__(len(sent))
        # but we share raw word features with our underlying sentence
        self._words = sent._words


def percentage(f):
    '''Decorate a function by modifying its return value into a percentage.

    The wrapped method must return a (numerator, denominator) ordered pair.
    '''
    def wrapper(*args, **kwargs):
        numer, denom = f(*args, **kwargs)
        if numer and denom:
            return 100.0 * numer / denom
        return 0.0
    return wrapper


def f1(f):
    '''Decorate a function by modifying its return value into an F1 score.

    The wrapped method must return a (precision, recall) ordered pair.
    '''
    def wrapper(*args, **kwargs):
        p, r = f(*args, **kwargs)
        if p and r:
            return 2 * p * r / (p + r)
        return 0.0
    return wrapper


class Evaluator(object):
    '''A stateful evaluator that can be used to assess parser accuracy.'''

    def __init__(self):
        '''Initialize this evaluator.'''
        # sentence accuracy
        self.total_sents = 0
        self.correct_sents = 0
        self.correctly_labeled_sents = 0

        # dependency accuracy
        self.total_deps = 0
        self.correct_deps = 0
        self.correctly_labeled_deps = 0

        # root P/R
        self.true_roots = 0
        self.correct_roots = 0
        self.predicted_roots = 0

        # dependency P/R by length
        self.true_length_deps = collections.defaultdict(int)
        self.correct_length_deps = collections.defaultdict(int)
        self.predicted_length_deps = collections.defaultdict(int)

        # label P/R
        self.true_labels = collections.defaultdict(int)
        self.correct_labels = collections.defaultdict(int)
        self.predicted_labels = collections.defaultdict(int)

    def measure(self, est, sent):
        '''Measure the accuracy of a parser estimate against a labeled sentence.

        est: An estimate that has been filled out by a parser.
        sent: The labeled source sentence.
        '''
        if len(est) != len(sent):
            logging.critical('Length of estimate %s != '
                             'length of sentence %s !', est, sent)
            sys.exit(1)

        all_correct = all_correct_labeled = True
        for mod in sent.itermods():
            # labeled and unlabeled accuracy
            self.total_deps += 1
            if est.head(mod) == sent.head(mod):
                self.correct_deps += 1
                if est.label(mod) == sent.label(mod):
                    self.correctly_labeled_deps += 1
                else:
                    all_correct_labeled = False
            else:
                all_correct = False

            # root P/R
            if sent.head(mod) == 0:
                self.true_roots += 1
            if est.head(mod) == 0:
                self.predicted_roots += 1
            if 0 == sent.head(mod) == est.head(mod):
                self.correct_roots += 1

            # label P/R
            if est.label(mod) == sent.label(mod):
                self.correct_labels[sent.label(mod)] += 1
            self.predicted_labels[est.label(mod)] += 1
            self.true_labels[sent.label(mod)] += 1

            # dependency P/R by length
            true_dist = abs(sent.head(mod) - mod)
            predicted_dist = abs(est.head(mod) - mod)
            if est.head(mod) == sent.head(mod):
                self.correct_length_deps[true_dist] += 1
            self.predicted_length_deps[predicted_dist] += 1
            self.true_length_deps[true_dist] += 1

        self.total_sents += 1
        if all_correct:
            self.correct_sents += 1
            if all_correct_labeled:
                self.correctly_labeled_sents += 1

    @percentage
    def unlabeled_sentence_score(self):
        return self.correct_sents, self.total_sents

    @percentage
    def labeled_sentence_score(self):
        return self.correctly_labeled_sents, self.total_sents

    @percentage
    def unlabeled_attachment_score(self):
        return self.correct_deps, self.total_deps

    @percentage
    def labeled_attachment_score(self):
        return self.correctly_labeled_deps, self.total_deps

    @percentage
    def root_precision(self):
        return self.correct_roots, self.predicted_roots

    @percentage
    def root_recall(self):
        return self.correct_roots, self.true_roots

    @f1
    def root_f1(self):
        return self.root_precision(), self.root_recall()

    @percentage
    def dependency_length_precision(self, length):
        return (self.correct_length_deps.get(length),
                self.predicted_length_deps.get(length))

    @percentage
    def dependency_length_recall(self, length):
        return (self.correct_length_deps.get(length),
                self.true_length_deps.get(length))

    @f1
    def dependency_length_f1(self, length):
        return (self.dependency_length_precision(length),
                self.dependency_length_recall(length))

    @percentage
    def label_precision(self, label):
        return (self.correct_labels.get(label),
                self.predicted_labels.get(label))

    @percentage
    def label_recall(self, label):
        return self.correct_labels.get(label), self.true_labels.get(label)

    @f1
    def label_f1(self, label):
        return self.label_precision(label), self.label_recall(label)

    def log_results(self):
        '''Display accuracy results in the logging output.'''
        logging.error('unlabeled trees: %.2f', self.unlabeled_sentence_score())
        logging.error('labeled trees: %.2f', self.labeled_sentence_score())
        logging.error('root P R F1: %.2f %.2f %.2f',
                      self.root_precision(),
                      self.root_recall(),
                      self.root_f1())

        logging.error('las: %.2f', self.labeled_attachment_score())
        logging.error('uas: %.2f', self.unlabeled_attachment_score())
        lengths = set(self.predicted_length_deps) | set(self.true_length_deps)
        for length in xrange(1, max(lengths) + 1):
            logging.error('dependency P R F1: %2d: %8d/%8d: %7.2f %7.2f %7.2f',
                          length,
                          self.predicted_length_deps.get(length, 0),
                          self.true_length_deps.get(length, 0),
                          self.dependency_length_precision(length),
                          self.dependency_length_recall(length),
                          self.dependency_length_f1(length))

        labels = set(self.predicted_labels) | set(self.true_labels)
        format = 'label P R F1: %%%ds: %%8d/%%8d: %%7.2f %%7.2f %%7.2f' % \
            max(len(l) for l in labels)
        for label in sorted(labels):
            logging.error(format,
                          label,
                          self.predicted_labels.get(label, 0),
                          self.true_labels.get(label, 0),
                          self.label_precision(label),
                          self.label_recall(label),
                          self.label_f1(label))
