#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
compare output of two methods for statistical significance

assumes reports have been dumped for both methods on same set of texts

TODO: list of args, matrix of results
"""

import cPickle
from pprint import pprint
import sys
from decoding import Report

eval1=cPickle.load(open(sys.argv[1]))
eval2=cPickle.load(open(sys.argv[2]))

args = sys.argv[1:]
MEASURE="F1"

def comp_matrix(all,measure="F1"):
    result = {}
    for i,one in enumerate(all[:-1]):
        #result[one]={}
        for other in all[i+1:]:
            eval1=cPickle.load(open(one))
            eval2=cPickle.load(open(other))
            result[one,other] = eval1.significance(eval2,measure=MEASURE,test="all")["wilcoxon"]
    return result


#if len(sys.argv)>3:
#    MEASURE=sys.argv[3]
if len(args)<=3:
    r= eval1.significance(eval2,measure=MEASURE,test="all")
    pprint(r)
    print "standard errors on measures:"
    print "%s : \t %1.3f %%" % (sys.argv[1],eval1.standard_error(measure=MEASURE)*100)
    print "%s : \t %1.3f %%" % (sys.argv[2],eval2.standard_error(measure=MEASURE)*100)

    print "confidence interval :"
    m1, (a1,b1) = eval1.confidence_interval(measure=MEASURE)
    m2, (a2,b2) = eval2.confidence_interval(measure=MEASURE)
    print "%s : \t %1.3f +- %f" % (sys.argv[1],m1*100,100*(m1-a1))
    print "%s : \t %1.3f +- %f" % (sys.argv[2],m2*100,100*(m2-a2))
else:
    res = comp_matrix(args)
    pprint(["%s %1.3e"%x for x in res.items() if x[1]<0.01])
