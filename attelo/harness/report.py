"""
Helpers for result reporting
"""

# Author: Eric Kow
# License: CeCILL-B (French BSD3-like)

# pylint: disable=too-few-public-methods


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
