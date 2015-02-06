"""
Helpers for result reporting
"""

# Author: Eric Kow
# License: CeCILL-B (French BSD3-like)

import csv

from attelo.cmd.report import EXPECTED_KEYS

# pylint: disable=too-few-public-methods


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


def mk_index(folds, configurations):
    """
    Generate an index file for attelo report (see :doc:`report`)

    :param folds: mapping of folds to relative path for data
                  associated with that fold (eg. `{3:'fold-3'}`)
    :type folds: [(int, string)]

    :param configurations: mapping of configurations to relative
                           path to outputs for that config
    :type configurations: [(:py:class:`EvaluationConfig`, string)]

    :rtype: dict (for json)
    """
    res = {'folds': [],
           'configurations': []}

    for fold_num, path in folds:
        res['folds'].append({'number': fold_num,
                             'path':  path})
    for config, path in configurations:
        res['configurations'].append(config.for_json(path))
    return res
