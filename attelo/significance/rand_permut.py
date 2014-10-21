import sys
from numpy.random import permutation
from scipy.stats import tstd, tmean
from math import sqrt

try:
    from scipy.misc import factorial
except: 
    print >> sys.stderr, "no scipy module, only approximate random permutation test possible"


def effect(data1,data2):
    s_pool = sqrt((tstd(data1)**2+tstd(data2)**2)/2.)
    return float(tmean(data1)-tmean(data2))/s_pool


def randperm_test(data1,data2,r=None,evalfunc=sum):
    """random permutation test to evaluate significance of
    difference between two data series of same length (n). 
    do r permutations of the concatenated data, compare scores of first and 2nd part
    the prop of times f(1st part)>f(data1) is the significance level for data1>data2
    
    this methos is NEVER exact, but is robust wrt the size of the data

    * r (int) : nb of permutations to do before stopping
        if None, do the nb of total permutations n! (warning : don't do it!)
    * evalfunc (iterable -> float): function apply for the evaluation on the data series
      defaut is to sum all values. 
    """
    n = len(data1)
    assert len(data1)==len(data2)
    if r is None:
        r = factorial(n,exact=False)
    # since r can be an approximation of a factorial, can be float -> while loop
    i = 0
    value = 0
    ref = evalfunc(data1)
    while i<r:
        i = i+1
        test = permutation(data1+data2)
        score = evalfunc(test[:n])
        if ref>score:
            value = value + 1
    return float(value)/r


if __name__=="__main__":
    import sys
    if len(sys.argv)>1:
        r = int(sys.argv[1])
    else:
        r = 1000
    data1= [1,1,1,0,0]*5
    data2= [1,0,0,0,0]*5
    p = 1- randperm_test(data1,data2,r)
    print "p, 1-p = ", p, 1-p
    print "effect = ", effect(data1,data2)
