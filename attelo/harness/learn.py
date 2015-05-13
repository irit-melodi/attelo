# Author: Eric Kow
# License: CeCILL-B (French BSD3-like)


"""
Test harness support for building models
"""

from __future__ import print_function
import os
import sys

from attelo.fold import (select_training)


def learn(hconf, econf, dconf, fold):
    """
    Run the learners for the given configuration
    """
    if fold is None:
        subpacks = dconf.pack
        parent_dir = hconf.combined_dir_path()
    else:
        subpacks = select_training(dconf.pack, dconf.folds, fold)
        parent_dir = hconf.fold_dir_path(fold)

    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    cache = hconf.model_paths(econf.learner, fold)
    print('learning ', econf.key, '...', file=sys.stderr)
    dpacks = subpacks.values()
    targets = [d.target for d in dpacks]
    econf.parser.payload.fit(dpacks, targets, cache=cache)


def mk_combined_models(hconf, econfs, dconf):
    """
    Create global for all learners
    """
    for econf in econfs:
        learn(hconf, econf, dconf, None)
