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

'''A non-projective, stack-based dependency parser.

This parser is based on Joakim Nivre's 2009 ACL paper, "Non-Projective
Dependency Parsing in Expected Linear Time." The base parser uses three actions
to attain a worst-case O(n) projective dependency parse ; to get non-projective
parses he adds a fourth action, SWAP. This action extends the runtime to at most
O(n**2), but in practice Nivre reports O(n) in most cases. Accuracy is
competitive with current state-of-the-art dependency parsers.
'''

# parser actions
SHIFT = 'SHIFT'
SWAP = 'SWAP'
LEFT_ARC = 'LEFT_ARC'
RIGHT_ARC = 'RIGHT_ARC'


class State(object):
    '''A parser state is a stack, a word index list, and a set of edges.

    States also have a reference to the sentence that contains the actual words
    from the sentence being parsed (the offsets in words, stack, and edges point
    into the list of words in the sentence).

    Finally, if a state was generated as part of a parse process from some other
    state, the previous state is linked, along with the score for the current
    state. The score for a parse is defined as the sum of the scores of each
    state in the parse.
    '''

    def __init__(self, est, words, stack, edges, score, prev):
        self.est = est
        self.words = words
        self.stack = stack
        self.edges = edges
        self.score = score
        self.prev = prev

    def __cmp__(self, other):
        return cmp(self.score, other.score)

    def can_shift(self):
        return self.words and self.words[0] != 0

    def shift(self, score=0):
        return State(self.est,
                     self.words[1:],
                     self.stack + self.words[0],
                     self.edges,
                     self.score + score,
                     self)

    def can_arc(self):
        return len(self.stack) > 1 and self.stack[-2] != 0

    def left_arc(self, score=0, label=None):
        i, j = self.stack[-2:]
        return State(self.est,
                     self.words,
                     self.stack[:-2] + (j, ),
                     self.edges + ((j, label, i), ),
                     self.score + score,
                     self)

    def right_arc(self, score=0, label=None):
        i, j = self.stack[-2:]
        return State(self.est,
                     self.words,
                     self.stack[:-2] + (i, ),
                     self.edges + ((i, label, j), ),
                     self.score + score,
                     self)

    def can_swap(self):
        return len(self.stack) > 1 and 0 < self.stack[-2] < self.stack[-1]

    def swap(self, score=0):
        i, j = self.stack[-2:]
        return State(self.est,
                     (i, ) + self.words,
                     self.stack[:-2] + (j, ),
                     self.edges,
                     self.score + score,
                     self)

    def execute(self, action, score=0, label=None):
        if action is SWAP:
            return self.swap(score)
        if action is SHIFT:
            return self.shift(score)
        if action is LEFT_ARC:
            return self.left_arc(score, label)
        if action is RIGHT_ARC:
            return self.right_arc(score, label)
        raise ValueError('unknown action %r' % action)

    def possible_actions(self):
        if self.can_shift():
            yield SHIFT
        if self.can_swap():
            yield SWAP
        if self.can_arc():
            yield LEFT_ARC
            yield RIGHT_ARC

    def is_terminal(self):
        return self.words == () and self.stack == (0, )


class Estimate(parser.Estimate):
    '''The estimate for a stack parser consists of a beam of current states.'''

    def __init__(self, sent, beam_width):
        '''
        '''
        super(Estimate, self).__init__(sent)
        self.beam_width = beam_width
        self.beam = [State(est,
                           range(1, len(self)),
                           (0, ),
                           (),
                           0,
                           None)]


def oracle(state, sentence):
    a = 0
    b = 2
    edges = ((i, l, j) for i, l, j in state.edges if i == a or j == b)
    if XXX: # TODO
        _, label, _ = edges.pop()
        return RIGHT_ARC, label
    if XXX: # TODO
        _, label, _ = edges.pop()
        return RIGHT_ARC, label
    if XXX: # TODO
        return SWAP, None
    return SHIFT, None


class Parser(parser.Parser):
    '''
    '''

    def __init__(self, extract, classifier):
        '''Set up this parser with a feature extractor and a classifier.'''
        self.extract = extract
        self.classifier = classifier

    def greedy(self, sentence):
        '''Iterate over a series of parser states for the given sentence.'''
        state = State(len(sentence))
        while True:
            yield state
            features = self.extract(state, sentence)
            actions = state.possible_actions()
            action, label, score = self.classifier.best(features, actions)
            state = state.execute(action, score=score, label=label)

    def parse(self, sent, **kwargs):
        '''Parse a single sentence.

        sent: A sentence to parse.

        Return the resulting parser estimate. Expected keyword arguments are:

        beam_width: The beam width for parsing. We maintain this-many active
          parser states during the parse, and discard states with low scores
          as we gather new ones with higher scores.
        '''
        est = Estimate(sent, kwargs.get('beam_width', 1))
        while est.has_incomplete_states():
            state = est.pop_state()
            features = self._ruleset.extract(state, sentence)
            actions = state.possible_actions()
            for a, l, s in self.classifier.ranked(features, actions):
                next = state.execute(a, label=l, score=s)
                if len(states) < beam_width - 1:
                    states.append(next)
                    continue
                if next.score < states[0].score:
                    break
                states.pop(0)
                bisect.insort(states, next)
        return est

    def learn(self, sent, action_classifier, label_classifier, **kwargs):
        '''Learn from the data in a single labeled sentence.

        sent: A labeled training sentence.
        action_classifier: The classifier to train for dependency actions.
        label_classifier: The classifier to train for dependency labels.
        '''
        state = State(len(sentence))
        self.classifier.train(features, action, label)
