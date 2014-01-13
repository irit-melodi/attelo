#!/usr/bin/env python

import sys
import os
import codecs
from add_feature import FeatureMap
from collections import defaultdict

DEF_ENCODING="utf-8"
ANA_NODE="m#NextNode"
ANTE_NODE="m#CandNode"
CLASS_ATTR="c#CLASS"
IN_SUFFIX=".features"

test_dir = sys.argv[1]
test_files = [os.path.join(test_dir,f) for f in os.listdir(test_dir)
              if f.endswith(IN_SUFFIX)]
print >> sys.stderr, "Nber test files: %s." %len(test_files)




correct1 = 0.0 # including multiple correct cases
total1 = 0

correct2 = 0.0 # not including them
total2 = 0

for tf in test_files:
    print >> sys.stderr, tf
    feat_map = FeatureMap(tf, encoding=DEF_ENCODING)
    correct_attachments = defaultdict(list)
    last_attachments = {}
    for instance in feat_map._all:
        edu2attach = instance.pop( ANA_NODE )
        attach_point = instance.pop( ANTE_NODE )
        cl = int(eval(instance.pop( CLASS_ATTR )))
        if cl == 1:
            correct_attachments[edu2attach].append( attach_point )
        is_last = eval(instance.get( "D#LAST", "False" ))
        if is_last:
            last_attachments[edu2attach] = attach_point
        # print edu2attach, attach_point, cl, is_last

    for edu, pts in correct_attachments.items():
        try:
            last = last_attachments[edu]
            total1 += 1
            if len(pts) == 1:
                total2 += 1
            if last in pts:
                correct1 += 1
                if len(pts) == 1:
                    correct2 += 1
        except KeyError:
            print >> sys.stderr, "No LAST for EDU %s! Skipping from ACC counts." %edu
        

print "ACC1: %s (%s/%s)" %(round(correct1/total1,2),correct1,total1)
print "ACC2: %s (%s/%s)" %(round(correct2/total2,2),correct2,total2)

