'''
Group-aware n-fold evaluation.

Attelo uses a variant of n-fold evaluation, where we (still)
andomly partition the dataset into a set of folds of roughly even
size, but respecting the additional constraint that any two data
entries belonging in the same "group" (determined a single
distiguished feature, eg. the document id, the dialogue id, etc)
are always in the same fold. Note that this makes it a bit harder
to have perfectly evenly sized folds


Created on Jun 20, 2012

@author: stergos

contribs: phil
'''

import random

from .edu import FAKE_ROOT_ID

def make_n_fold(dpack, folds, rng):
    """Given a data pack and a desired number of folds, return a fold selection
    dictionary assigning a fold number to each each grouping (see
    :py:class:attelo.edu.EDU:).

    :type dpack: :py:class:DataPack:
    :type folds: int
    :param rng: random number generator (hint: the random module
                will be just fine if you don't mind shared state)
    :type rng: :py:class:random.Random:

    :rtype dict(string, int)
    """
    if rng is None:
        rng = random
    groupings = list(set(x.grouping for x in dpack.edus
                         if x.id != FAKE_ROOT_ID))

    if folds < 2:
        raise ValueError("Must have more than 1 fold")
    elif len(groupings) < folds:
        oops = ("Too many folds: I can't make {folds} folds when "
                "there are only {groupings} distinct edu groupings")
        raise ValueError(oops.format(folds=folds,
                                     groupings=len(groupings)))

    fold_dict = {}
    # to divide $g groupings into $f folds (typically
    # $g > $f), we work one $f-sized block at a time
    # randomly/exhaustively assigning each fold to
    # one of the groups within the block
    #
    # it's ok if our last block has fewer groupings
    # than we need.
    blocks = (len(groupings) / folds) + 1
    for current in xrange(blocks):
        random_values = rng.sample(xrange(folds), folds)
        for i in xrange(folds):
            position = (current * folds) + i
            if position < len(groupings):
                grp = groupings[position]
                fold_dict[grp] = random_values[i]
    return fold_dict


def fold_groupings(fold_dict, fold):
    '''
    Return the set of groupings that belong in a fold.
    Raise an exception if the fold is not in the fold dictionary

    :rtype frozenset(int)
    '''
    res = []
    for group, gfold in fold_dict.items():
        if gfold == fold:
            res.append(group)
    if not res:
        oops = 'There is no fold "{f}" in the dictionary {d}'
        raise ValueError(oops.format(f=fold, d=fold_dict))
    return frozenset(res)
