'''
Random permutation test (see :py:meth:randperm_test:)
'''

from __future__ import print_function

import sys
from numpy.random import permutation
from scipy.stats import tstd, tmean
from math import sqrt

try:
    from scipy.misc import factorial
except ImportError:
    print("no scipy module, only approximate random permutation test possible",
          file=sys.stderr)


def effect(data1, data2):
    s_pool = sqrt((tstd(data1)**2 +
                   tstd(data2)**2)
                  /2.)
    return float(tmean(data1)-tmean(data2))/s_pool


def randperm_test(data1, data2, perms=None, evalfunc=sum):
    """random permutation test to evaluate significance of
    difference between two data series of same length (n).
    do r permutations of the concatenated data, compare scores of first and 2nd part
    the prop of times f(1st part)>f(data1) is the significance level for data1>data2

    this methos is NEVER exact, but is robust wrt the size of the data

    :param:perms: number of permutations to do before stopping
                  if None, do the nb of total permutations n!
                  (warning : don't do it!)
    :type:perms: int or None
    :param evalfunc: function apply for the evaluation on the data series
                     default is to sum all values.
    :type:iterations: iterable -> float
    """
    length = len(data1)
    assert len(data1) == len(data2)
    if perms is None:
        perms = factorial(length, exact=False)
    # since r can be an approximation of a factorial, can be float -> while loop
    i = 0
    value = 0
    ref = evalfunc(data1)
    while i < perms:
        i = i + 1
        test = permutation(data1+data2)
        score = evalfunc(test[:length])
        if ref > score:
            value = value + 1
    return float(value)/perms


# pylint: disable=invalid-name
def _self_test():
    'Check that we are doing something reasonable'
    if len(sys.argv) > 1:
        perms = int(sys.argv[1])
    else:
        perms = 1000
    data1 = [1, 1, 1, 0, 0] * 5
    data2 = [1, 0, 0, 0, 0] * 5
    p = 1 - randperm_test(data1, data2, perms)
    print("p, 1-p = ", p, 1-p)
    print("effect = ", effect(data1, data2))

if __name__ == "__main__":
    _self_test()
