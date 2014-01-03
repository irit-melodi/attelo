#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
count number of occurrences of polka feature file
"""

import sys
import os
import codecs
from polka.common.datasource import ClassificationSource 
from polka.common.datasource import RankingSource


CLASS_SUFFIX="class"
RANK_SUFFIX="rank"

if __name__ == "__main__":

    in_file = sys.argv[1]
    cts = {}
    if in_file.endswith(CLASS_SUFFIX):
        data = ClassificationSource(in_file)
        # print data.size()
        for k,inst in enumerate(data.get_input()):
            os.write(1, "%s" %"\b"*len(str(k))+str(k))
            _,fv = inst
            for f,v in fv:
                cts[f] = cts.get(f,0) + 1
    elif in_file.endswith(RANK_SUFFIX):
        data = RankingSource(in_file)
        for k,inst in enumerate(data.get_input()):
            os.write(1, "%s" %"\b"*len(str(k))+str(k))
            for _,cand_set in inst:
                for cand in cand_set:
                    for f,v in cand:
                        cts[f] = cts.get(f,0) + 1
    cts = cts.items()        
    cts.sort(key=lambda x:x[1],reverse=True)

    for f,ct in cts:
        print "%s\t%s" %(f,ct)

