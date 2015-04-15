"""
Attelo is essentially a toolkit for producing parsers: parsers are black boxes
that take EDUS as inputs and produce graphs as output.

Parsers follow the scikit fit/transform idiom. They are learned from some
training data via the `fit()` function (this usually results in some model
that the parser remembers;, but a hypothetical purely rule-based parser might
have a no-op fit function). Once fitted to the training data, they can be set
loose on anything you might want to parse: the `transform` function will
produce graphs from the EDUs.
"""

from .interface import Parser
