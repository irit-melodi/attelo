#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
add temporal features to EDUs

in: xml annodis file + temporal gazetteer

out: same with additional <feature> objects signalling temporal markes

usage:
python timex_tag.py ../../../data/attachment/dev/ER187-Eponge.xml ../../config/temporal_markers/ 

for f in ../../../data/attachment/dev/*.xml ; do python timex_tag.py $f ../../config/temporal_markers/ ; done; 

TODO: so far, only lexical match for timex. pretty rough
   - type date/time
   - time deictic/anaphoric
   - tag signals too
   - real timex tagging !
   - matching segments is brute-force n^2 iteration. baaaad
     -> better to order matching on spans and iterate through them (sort is nlog n + retrive is linear)
     alt: interval tree indexing (but building is nlog n / retrieval is log n but used once only)
"""


import sys
import os
import os.path
import codecs
import re
import xml.etree.ElementTree as ET

from AnnodisReader import annodisAnnot, Preprocess
from Lookup import Gazetteer

def includes(pos1,pos2):
    a1,b1=pos1
    a2,b2=pos2
    return a1<=a2 and b2<=b1


if __name__=="__main__":
    try:
        doc=annodisAnnot(sys.argv[1]) 
    except:
        print "ERROR reading file:", sys.argv[1]
        sys.exit(0)
    try:
        prep=Preprocess(sys.argv[1].split(".xml")[0]+".txt.prep.xml")
        doc.add_preprocess(prep)
    except:
        print "ERROR reading prepocessed file for", sys.argv[1]
        sys.exit(0)


    lexicon=Gazetteer(sys.argv[2])

    txt=doc.text()
    lookup=lexicon.tag(txt)

    for one in doc.edus():
        all=[]
        span=int(one.attrib["start"]),int(one.attrib["end"])
        #print >> sys.stderr, "edu", span
        tokens=doc.get_edu_tokens(one.attrib["id"])
        txt = " ".join([x.lemma() for x in tokens])
        #print >> sys.stderr, txt
        verb_class= lexicon.tag(txt).get("verb_classes",[])
        if verb_class!=[]:
            verb_class=set([z.val() for (x,y,z) in verb_class])
            one.attrib["verb_class"]="+".join(verb_class)          
         


        for atype in lookup:
            for matched in lookup[atype]:
                pos=matched[0],matched[1]
                #print >> sys.stderr, pos
                if includes(span,pos):
                    #print >> sys.stderr, "matched", one.attrib["id"], matched
                    all.append((atype,matched[2]))
        if all!=[]:
            # all attributes
            #one.attrib["timex"]=",".join(["%s:%s"%(x,y.val()) for (x,y) in all])
            signals=[(x,y.val()) for (x,y) in all if x=="signals"]   
            #print >> sys.stderr, " ".join(alltypes)
            if signals!=[]:
                one.attrib["timex_signals"]="_".join([y for (x,y) in signals])
              # just a presence
            if len(all)>len(signals)+len(verb_class):
                one.attrib["timex"]="True"
    
            # todo: timex types: date/time/duration + deictic / anaphoric etc

    doc.save(doc.name()+".xml")
