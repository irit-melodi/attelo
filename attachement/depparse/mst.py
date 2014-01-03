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

'''A non-projective, maximum spanning tree (MST) dependency parser.

An MST parser operates by constructing a connected graph where each word in a
sentence is a node in the graph, and each directed edge is a labeled and scored
dependency relationship between two words in the sentence. Constructing the
graph is quadratic in the number of words in the sentence, since each possible
word pair must be scored for both dependency arc affinity, and (for arcs with
high scores) arc label affinity. Once the fully connected graph has been formed,
we use standard graph algorithms to find the MST that starts at the special ROOT
node. This MST is returned as the dependency structure for the sentence.

MST parsers are somewhat more expensive than the alternative, shift-reduce,
parsers, but they have a compellingly simple architecture, and there is
something interesting about the fully connected graph that gets constructed
during the parsing process.
'''

import re
import sys
import math
import random
import logging
import collections

import graph
import parser
import sentence


class Estimate(parser.Estimate):
    '''The estimate for MST parsers uses quadratic score and label arrays.'''

    def __init__(self, sent):
        '''Initialize this estimate with a sentence.'''
        super(Estimate, self).__init__(sent)
        self._max_scores = [-1e100] * len(self)
        self._qscores = {}
        self._qlabels = {}

    def set_link_score(self, mod, head, score, label):
        '''Set a score on a mod:head link in this estimate.'''
        if score > self._max_scores[mod]:
            self._max_scores[mod] = score
            self._heads[mod] = head
            self._labels[mod] = label
        self._qscores[head, mod] = score
        self._qlabels[head, mod] = label

    def apply_mst(self):
        '''Apply the MST to our estimates of sentence head and label data.'''
        succs = collections.defaultdict(list)
        for mod in self.itermods():
            for head in self.iterheads():
                if mod != head:
                    succs[mod]
                    succs[head].append(mod)
        floor = min(self._qscores.itervalues())
        score = lambda h, m: self._qscores[h, m] - floor
        label = lambda h, m: self._qlabels[h, m]
        for head, mod in graph.Digraph(succs, score, label).mst().iteredges():
            original_head = self.head(mod)
            if original_head != head:
                logging.debug('updating head for %s from %s[%s] to %s[%s]',
                              self.as_str(mod),
                              self.as_str(original_head),
                              self._qlabels[original_head, mod],
                              self.as_str(head),
                              self._qlabels[head, mod])
                self._heads[mod] = head
                self._labels[mod] = self._qlabels[head, mod]


MOTION_UP = 0
MOTION_DOWN = 1

class Rule(object):
    '''A Rule describes how to extract features from a sentence.'''

    def __init__(self, raw):
        '''Initialize with a raw rule description string.'''
        self.is_tree_feature = False

        names = []
        subrules = []
        for chunk in raw.split('+'):
            r, n = self._parse(chunk)
            subrules.append(r)
            names.append(n)
        self._subrules = subrules
        self._name = '+'.join(names)
        logging.warning('added feature rule %r', self._name)

    def _parse(self, chunk):
        '''Parse a chunk of a subrule string.'''
        tokens = filter(None, re.split(r'[\.\s\[\]\(\)]+', chunk.strip()))

        feature_token = tokens.pop()
        feature = self._parse_feature(feature_token)

        # extract starting position, offset, and tree motion words.
        start_from_head = None
        offset = None
        motions = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            i += 1
            if token == 'head':
                assert start_from_head is None
                start_from_head = True
                offset = int(tokens[i])
                i += 1
            elif token == 'mod':
                assert start_from_head is None
                start_from_head = False
                offset = int(tokens[i])
                i += 1
            elif token == 'up':
                motions.append(MOTION_UP)
                self.is_tree_feature = True
            elif token == 'down':
                motions.append(MOTION_DOWN)
                self.is_tree_feature = True
            else:
                raise ValueError('unknown rule token %r' % token)

        mh = 'mh'[start_from_head]
        ud = ''.join('ud'[m] for m in motions)
        off = offset and str(offset) or ''

        logging.debug('extracted subrule %s:%d:%s:%s', mh, offset, ud, feature)

        return ((start_from_head, offset, motions, feature),
                '%s%s%s.%s' % (mh, off, ud, feature_token[0]))

    def _parse_feature(self, token):
        '''Convert a feature name into a feature index.'''
        if token == 'deprel':
            self.is_tree_feature = True
            return sentence.FEATURE_LABEL
        if token in sentence.FEATURES:
            return sentence.FEATURES.index(token)
        raise NameError('unknown feature %r' % token)

    def apply(self, seq, mod, head):
        '''Apply this rule to an information source.

        seq: A Sequence for features and head/label information
        mod: The index of the candidate modifier word in the sentence
        head: The index of the candidate head word in the sentence

        Return a (name, value) pair containing the rule name and the extracted
        values.
        '''
        values = []
        for start_from_head, offset, motions, feature in self._subrules:
            o = [mod, head][start_from_head] + offset
            if not 0 <= o < len(seq):
                values.append('')
                continue
            offsets = [o]
            for motion in motions:
                if motion == MOTION_UP:
                    offsets = [seq.head(o) for o in offsets]
                elif motion == MOTION_DOWN:
                    _offsets = []
                    for o in offsets:
                        _offsets.extend(seq.children(o))
                    offsets = _offsets
                else:
                    raise ValueError('unknown motion %r' % motion)
            values.append('~'.join(seq.feature(o, feature) for o in offsets))
        return '%s:%s' % (self._name, '+'.join(values))


BIAS_FEATURE = '_'

class Ruleset(list):
    '''A ruleset is a list of feature extraction rules.

    Call apply() to apply the ruleset to a sequence, producing an iterator over
    feature strings.
    '''

    def __init__(self, filename, postag_window=5, log_distance=1.5):
        '''Load a rule set from the given filename.

        filename: The name of a file to read and parse for feature rules.
        postag_window: Include a feature for each word pair that contains at
          most this many POS tags between mod and head in each direction.
        log_distance: If not None, "bin" the distance between mod and head
          using log(d, this-base) ; if False, use the real linear distance.
        '''
        self.postag_window = postag_window
        self.log_distance = log_distance

        for i, l in enumerate(open(filename)):
            l = l.strip()
            if l.startswith('#') or l.startswith('//') or not l:
                continue
            try:
                self.append(Rule(l.lower()))
            except:
                logging.critical(
                    '%s line %d: cannot parse rule: %r', filename, i, l)
                sys.exit(1)

    def _apply(self, seq, mod, head, use_tree_features=True):
        '''Iterate over our rules to produce a feature set.'''
        def pos(start, inc, limit):
            for i in xrange(start, start + inc * self.postag_window, inc):
                if i < 0 or i >= len(seq):
                    break
                yield seq.feature(i, sentence.FEATURE_POSTAG)
                if i == limit:
                    break

        minc, hinc = ((1, -1), (-1, 1))[mod > head]
        yield 'mh.p:%s' % '+'.join(pos(mod, minc, head))
        yield 'hm.p:%s' % '+'.join(pos(head, hinc, mod))
        for rule in self:
            if use_tree_features or not rule.is_tree_feature:
                yield rule.apply(seq, mod, head)
        yield BIAS_FEATURE

    def apply(self, seq, mod, head, use_tree_features=True):
        '''Apply this Ruleset to a Sequence at a particular pair of words.

        seq: A Sequence to use for rule application.
        mod: The index of a modifier word in the Sequence.
        head: The index of a potential head word in the Sequence.
        use_tree_features: If True, apply Rules that rely on tree information.
          If False, do not apply these Rules.
        '''
        dist = abs(head - mod)
        if self.log_distance:
            dist = int(math.log(1 + dist, self.log_distance))
        for feature in self._apply(seq, mod, head, use_tree_features):
            yield '%c%s:%s' % ('-+'[head >= mod], dist, feature)
            if feature is not BIAS_FEATURE:
                yield feature
        yield BIAS_FEATURE


NONE_LABEL = '_'

class Parser(parser.Parser):
    '''This class parses each sentence using an MST estimate.'''

    def learn(self, sent, action_classifier, label_classifier, **kwargs):
        '''Learn from the data in a single labeled sentence.

        sent: A labeled training sentence.
        action_classifier: The classifier to train for dependency actions.
        label_classifier: The classifier to train for dependency labels.

        Expected keyword arguments are:

        negative_sample_ratio: Train using this many negative examples per
          positive example. For example, if this is 2, then 2 non-dependency
          examples will be passed to the classifier for each true dependency
          example. Higher values generally result in more accurate parsers, but
          take longer to train. Defaults to 5.
        '''
        # there are n(n+1) word pairs in a sentence of length n (the n+1 factor
        # includes ROOT as a potential head for each word). n of these are
        # positive examples that we will use for training. we want to pick out
        # some number of the remaining n**2 word pairs as negative training
        # examples.
        threshold = kwargs.get('negative_sample_ratio', 5.0) / (len(sent) - 1)
        for mod in sent.itermods():
            for head in sent.iterheads():
                attach = sent.head(mod) == head
                if attach or random.random() < threshold:
                    features = tuple(self._ruleset.apply(sent, mod, head))
                    action_classifier.train(features, attach)
                    if attach:
                        label_classifier.train(features, sent.label(mod))
                    logging.debug('%s -> %s [%s]',
                                  sent.as_str(mod),
                                  sent.as_str(head),
                                  attach and sent.label(mod) or NONE_LABEL)

    def iterparse(self, sent, max_improvements=0):
        '''Parse a single sentence, generating a sequence of estimates.

        sent: A sentence to parse.
        max_improvements: The number of improvement passes to make after the
          initial parse attempt.
        '''
        est = Estimate(sent)

        for i in xrange(1 + max_improvements):
            for mod in est.itermods():
                for head in est.iterheads():
                    features = tuple(self._ruleset.apply(est, mod, head, i > 0))
                    attach, score = self._action_classifier.score(features)
                    label = NONE_LABEL
                    if attach:
                        label, score = self._label_classifier.score(features)
                    else:
                        score = -score
                    est.set_link_score(mod, head, score, label)
                    logging.debug('%s -> %s [%s:%.2f]',
                                  sent.as_str(mod),
                                  sent.as_str(head),
                                  label,
                                  score)
                logging.debug('parsing pass %d: '
                              '%s attaches to %s with %s, '
                              'classifies as %s with %s',
                              i,
                              sent.as_str(mod),
                              sent.as_str(sent.head(mod)), sent.label(mod),
                              est.as_str(est.head(mod)), est.label(mod))

            # use the mst to improve our greedy attachment estimates by
            # performing a global analysis of the resulting graph.
            est.apply_mst()

            # pass back the current estimate
            yield est

    def parse(self, sent, **kwargs):
        '''Parse a single sentence.

        sent: A sentence to parse.

        Return the resulting parser estimate. Expected keyword arguments are:

        max_improvements: The maximum number of tree-based passes to make after
          the initial parsing pass. Defaults to 0.
        '''
        est = None
        for est in self.iterparse(sent, kwargs.get('max_improvements', 0)):
            pass
        return est
