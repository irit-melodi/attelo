"""
inter annotator agreement for discourse markers use

usage
prog annot1_directory annot2_directory [plain]

last optional argument can be anything, if set means ignore relation type, just 
distinguish between none/all others

then, can be fed to kappa script:
cut -f 2-3 test | python code/src/comparison/kappa.py 
(TODO: all in one)
"""

import sys
import re
from collections import defaultdict
from AnnodisReader import *
from markers import LexConn
from datetime import datetime
import os.path
import codecs

corpus1 = annodisCorpus(sys.argv[1],format="glozz")
corpus2 = annodisCorpus(sys.argv[2],format="glozz")
corpus1.add_markers()
corpus2.add_markers()

if len(sys.argv)==4:
    plain = True
else:
    plain = False

# if "None" in annotation 1  means discourse usage
SPECIAL = True

for name,doc1 in corpus1.docs().items():
    dirname, filename = os.path.split(name)
    name2 = os.path.join(sys.argv[2],filename)
    doc2 = corpus2.docs().get(name2)
    # because there is no control on filename types from annodis reader... 
    filename = filename.decode("utf-8")
    if doc2:
        all1 = dict([(doc1.get_marker_span(id),marker["attrib"]["relation"]) for id,marker in doc1.markers().items()])
        all2 = dict([(doc2.get_marker_span(id),marker["attrib"]["relation"]) for id,marker in doc2.markers().items()])
        for ((s,e),rel1) in all1.items():
            # process rel name
            if rel1 is None or rel1=="altlex":
                if SPECIAL:
                    rel1 = "discourse"
                else:
                    rel1 = "none"
            else:
                rel1 = rel1.split(":")[0]
                if plain and rel1!="none":
                    rel1 = "discourse"
            rel2 = all2.get((s,e),"none")
            rel2="none" if rel2 is None else rel2.split(":")[0]
            if rel2!="none" and plain: rel2="discourse"
            print (u"%s-%s-%s\t%s\t%s"%(filename,s,e,rel1,rel2)).encode("utf-8")
