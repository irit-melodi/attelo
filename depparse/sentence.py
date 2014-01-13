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

'''A basic Sentence class and related functions for dependency parsing.'''

import random


# The features that we allow from our raw training (and test) data.
FEATURE_FORM = 0
FEATURE_LEMMA = 1
FEATURE_POSTAG = 2
FEATURES = ('form', 'lemma', 'postag')

FEATURE_LABEL = None

# Every Sentence starts with this set of ROOT word features.
ROOT = ('<S>', '<S>', '<S>')


class Sequence(object):
    '''A high-level base class for Sentence and State objects.'''

    def __init__(self, length):
        '''Initialize a Sequence of a particular length.'''
        self._length = length
        self._words = [None] * length
        self._heads = [0] * length
        self._labels = ['_'] * length
        self._strs = [''] * length

    def __len__(self):
        '''Get the length of the Sequence.'''
        return self._length

    def __str__(self):
        '''Get a string showing the words in the Sequence.'''
        return ' '.join(self.as_str(w) for w in self.itermods())

    def as_postscript(self, height=20):
        '''Return a PostScript string showing the tree for this sequence.'''
        lines = ['/w (ROOT) stringwidth pop def',
                 '/xl 0 def',
                 '/xc0 w 2 div def',
                 '0 0 moveto',
                 '(ROOT) show']

        # show all the words in the sentence, in addition to calculating the
        # x-offsets of the centers of each word.
        for mod in self.itermods():
            s = self.feature(mod, FEATURE_FORM)
            form = s.replace(')', '').replace('(', '')
            s = self.feature(mod, FEATURE_POSTAG)
            postag = s.replace(')', '').replace('(', '')
            lines.extend((
                    '/w (%s/%s) stringwidth pop def' % (form, postag),
                    '/xl xl w %d add add def' % height,
                    '/xc%d xl w 2 div add def' % mod,
                    'xl 0 moveto',
                    '(%s/%s) show' % (form, postag)))

        # now add the dependency arcs. for each word in the sentence, draw an
        # arc label, and then draw the arc from the left to the right.
        for mod in self.itermods():
            head = self.head(mod)
            y = height + 20 * abs(mod - head)
            dist = max(10 - 2 * abs(mod - head), 0)
            lines.extend((
                    # label
                    '/hl (%s) stringwidth pop 2 div def' % self.label(mod),
                    '/ha xc%d xc%d sub 2 div def' % (head, mod),
                    'xc%d ha add hl sub %d 0.78 mul moveto' % (mod, y),
                    '(%s) show' % self.label(mod),
                    # arc
                    '/xl xc%d %d add def' % (min(mod, head), dist),
                    '/xr xc%d %d sub def' % (max(mod, head), dist),
                    'xl %d moveto' % height,
                    'xl %d xr %d xr %d curveto' % (y, y, height),
                    'stroke',
                    # arrow
                    '/xl x%c def' % 'lr'[head < mod],
                    'xl %d moveto' % height,
                    'xl 2 add %d 5 add lineto' % height,
                    'xl 2 sub %d 5 add lineto' % height,
                    'xl %d lineto' % height,
                    'fill'))

        return '\n'.join(lines)

    def as_dot(self):
        '''Return a dot (graphviz) string showing the tree for this sentence.'''
        lines = ['digraph sentence {', '_0 [label="ROOT"];']
        for mod in self.itermods():
            name = '%d\\n%s\\n%s' % (
                mod,
                self.feature(mod, FEATURE_FORM).replace('"', 'Q'),
                self.feature(mod, FEATURE_POSTAG))
            lines.append('_%d [label="%s"];' % (mod, name))
            label = self.label(mod).replace('"', 'Q')
            lines.append('_%d -> _%d [label="%s"];' %
                         (mod, self.head(mod), label))
        lines.append('}')
        return '\n'.join(lines)

    def as_str(self, mod):
        '''Get a string showing a word in the Sequence.'''
        if not self._strs[mod]:
            self._strs[mod] = '%d|%s|%s' % (mod,
                                            self.feature(mod, FEATURE_FORM),
                                            self.feature(mod, FEATURE_POSTAG))
        return self._strs[mod]

    def head(self, mod):
        '''Get the head for a particular word in the Sequence.'''
        return self._heads[mod]

    def children(self, head):
        '''Get the children for a head in the Sequence.'''
        return (mod for mod in self.itermods() if self.head(mod) == head)

    def label(self, mod):
        '''Get the label for a particular word in the Sequence.'''
        return self._labels[mod]

    def feature(self, mod, feature):
        '''Get a feature from a particular word in the Sequence.'''
        if feature is FEATURE_LABEL:
            return self._labels[mod]
        return self._words[mod][feature]

    def itermods(self):
        '''Iterate over all valid modifier indices in the Sequence.'''
        for mod in xrange(1, len(self)):
            yield mod

    def iterheads(self):
        '''Iterate over all valid head indices in the Sequence.'''
        for head in xrange(0, len(self)):
            yield head


class Sentence(Sequence):
    '''A Sentence is just a raw Sequence of words.'''

    def __init__(self, words):
        super(Sentence, self).__init__(1 + len(words))
        self._words[0] = ROOT
        for i, word in enumerate(words):
            self._words[i+1] = tuple(word.get(f, '_') for f in FEATURES)
            self._heads[i+1] = int(word.get('head', 0))
            self._labels[i+1] = word.get('deprel', '_')


# These are the labels on the columns in the CoNLL 2009 dataset.
CONLL_COLUMNS = ('id',
                 'form',
                 'lemma', 'plemma',
                 'postag', 'ppostag',
                 'feats', 'pfeats',
                 'head', 'phead',
                 'deprel', 'pdeprel',
                 'fillpred',
                 'sense',
                 )

def words_from_conll(lines):
    '''Read words for a single sentence from a CoNLL text file.'''
    return [dict(zip(CONLL_COLUMNS, line.split('\t'))) for line in lines]


def lines_from_conll(lines):
    '''Read lines for a single sentence from a CoNLL text file.'''
    for line in lines:
        if not line.strip():
            return
        yield line.strip()


def sentences_from_conll(handle, count=None):
    '''Read at most count sentences from lines in an open CoNLL file handle.'''
    reservoir = []
    while True:
        lines = tuple(lines_from_conll(handle))
        if not len(lines):
            break
        reservoir.append(lines)
    if count and len(reservoir) > count:
        reservoir = random.sample(reservoir, count)
    return [Sentence(words_from_conll(r)) for r in reservoir]
