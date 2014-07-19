"""
Feature vectors
"""

# Author: Eric Kow <eric@erickow.com>
# License: CeCILL-B (French BSD3, but see caveat in README)

from collections import namedtuple

_Phrasebook = namedtuple("Phrasebook",
                         ["source",
                          "target",
                          "target_span_start",
                          "target_span_end",
                          "source_span_start",
                          "source_span_end",
                          "grouping",
                          "label"])


class Phrasebook(_Phrasebook):
    """
    Distinguished feature names. Any attelo feature vector must
    contain at least these features (but they can have any name)
    ::

    * for a source and a target node:
        * its id
        * its text-span start
        * its text-span end
    * a label
    * some grouping of nodes into larger units (for example,
      the nodes for annotations in the same file)

    The phrasebook tells attelo what are the names for these key
    features.
    """
    pass
