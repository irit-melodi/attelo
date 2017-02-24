"""Parser made by sequencing other parsers.

Ideally, we'd like to use sklearn.pipeline.Pipeline but our previous
attempts have failed.
The current trend is to try and slowly converge.
"""

from __future__ import absolute_import, print_function

from .interface import Parser


class Pipeline(Parser):
    """Apply a sequence of parsers.

    NB. For now we assume that these parsers can be
    fitted independently of each other.

    Steps should be a tuple of names and parsers, just like
    in sklearn.

    Parameters
    ----------
    steps : list
        List of (name, parser) tuples that are chained.

    Attributes
    ----------
    named_steps : dict
        Read-only attribute to access any step parameter by user given
        name. Keys are step names and values are step parameters.
    """

    # BaseEstimator interface (wip)

    def __init__(self, steps):
        self.steps = list(steps)
        self._validate_steps()
        # missing: memory ;
        # we already have 'cache' in fit() and transform() but 'memory'
        # might be a better replacement (?)

    def _validate_names(self, names):
        """simplified from _BasePipeline._validate_names.

        We don't have get_params yet.
        """
        if len(set(names)) != len(names):
            raise ValueError('Names provided are not unique: '
                             '{0!r}'.format(list(names)))
        # missing:
        # * step names that conflict with constructor arguments ;
        # * step names containing '__'

    def _validate_steps(self):
        """simplified from Pipeline._validate_steps.

        missing several required parts of the API, including get_params
        and requirements on all chained estimators and the last one
        """
        names, parsers = zip(*self.steps)
        # validate names
        self._validate_names(names)
        # missing: validate parsers, check their API

    @property
    def named_steps(self):
        return dict(self.steps)

    def fit(self, dpacks, targets, nonfixed_pairs=None, cache=None):
        """Fit."""
        for name, parser in self.steps:
            parser.fit(dpacks, targets, nonfixed_pairs=nonfixed_pairs,
                       cache=cache)

    def transform(self, dpack, nonfixed_pairs=None):
        """Transform."""
        for name, parser in self.steps:
            dpack = parser.transform(dpack, nonfixed_pairs=nonfixed_pairs)
        return dpack
