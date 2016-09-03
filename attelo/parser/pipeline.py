"""
Parser made by sequencing other parsers
"""

from __future__ import print_function

# FIXME: look into using sklearn.pipeline.Pipeline
# I wasn't too successful last time

from .interface import Parser


class Pipeline(Parser):
    """
    Apply a sequence of parsers.

    NB. For now we assume that these parsers can be
    fitted independently of each other

    Steps should be a tuple of names and parsers, just like
    in sklearn.pipeline.Pipeline.
    """
    def __init__(self, steps):
        self.steps = steps

    def fit(self, dpacks, targets, nonfixed_pairs=None, cache=None):
        for name, parser in self.steps:
            print('Pipeline: fit ', name)
            parser.fit(dpacks, targets, nonfixed_pairs=nonfixed_pairs,
                       cache=cache)
            print('... done')

    def transform(self, dpack, nonfixed_pairs=None):
        for name, parser in self.steps:
            # print('Pipeline: transform ', name)
            dpack = parser.transform(dpack, nonfixed_pairs=nonfixed_pairs)
            # print('... done')
        return dpack
