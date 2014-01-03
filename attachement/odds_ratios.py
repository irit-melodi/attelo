#!/usr/bin/env python

"""

odds(f_i,y1,y2) = P(f_i=1|y1) / P(f_i=1|y2)

"""

import sys
import os
import codecs
from add_feature import FeatureMap

DEF_ENCODING="utf-8"
ANA_NODE="NextNode"
ANTE_NODE="CandNode"
CLASS_ATTR="ATTACHMENT"
IN_SUFFIX=".features"

in_dir = sys.argv[1]
in_files = [os.path.join(in_dir,f) for f in os.listdir(in_dir)
            if f.endswith(IN_SUFFIX)]
print >> sys.stderr, "Nber in files: %s." %len(in_files)


# collect counts
f_1_counts = {}
f_0_counts = {}

for tf in in_files:
    print >> sys.stderr, tf
    feat_map = FeatureMap(tf, encoding=DEF_ENCODING)
    for instance in feat_map._all:
        edu2attach = instance.pop( ANA_NODE )
        attach_point = instance.pop( ANTE_NODE )
        cl = int(eval(instance.pop( CLASS_ATTR )))
        d2update = f_0_counts
        if cl == 1:
            d2update = f_1_counts
        for fv_pair in instance.items():
            d2update[fv_pair] = d2update.get(fv_pair,0) + 1

# compute odds
odds = {}

all_feats = f_1_counts.keys() # + f_0_counts.keys()

for f in all_feats:
    f_1_ct = f_1_counts[f]
    if f not in f_0_counts:
        # plus-1 smoothing
        f_1_ct += 1
        f_0_ct = 1
    else:
        f_0_ct = f_0_counts[f] 
    odds[f] = f_1_ct / float( f_0_ct )

odds = odds.items()
odds.sort( key=lambda x:x[1], reverse=True )

for f,r in odds:
    print f, "\t", r


    
