"""
Helpers for result reporting
"""

# Author: Eric Kow
# License: CeCILL-B (French BSD3-like)

import csv

from attelo.cmd.report import EXPECTED_KEYS


# pylint: disable=pointless-string-statement, too-few-public-methods
class CountIndex(object):
    """
    Convenience wrapper to generate attelo count index files.
    Any exceptions raised will be bubbled up

    with CountIndex(path) as writer:
        writer.writerow(foo)

    """
    def __init__(self, path):
        self.path = path
        "path to the wrapped count file"

        self._stream = None
        self._writer = None

    def __enter__(self):
        self._stream = open(self.path, "w")
        self._writer = csv.DictWriter(self._stream,
                                      fieldnames=EXPECTED_KEYS)
        self._writer.writeheader()
        return self._writer

    def __exit__(self, etype, value, tback):
        if tback is None:
            self._stream.close()
# pylint: enable=pointless-string-statement, too-few-public-methods
