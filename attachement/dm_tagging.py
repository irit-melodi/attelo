#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
wip: redo the discourse marker projection on glozz annodis files
input: glozz manual annotation, a lexconn-like DM resource
output: new glozz file with discourse markers added

input should be in 
 $Annodis/data/annot_experts/glozz

usage: 
prog dir-with-annotations [lexconn-database]

output is in current directory

TODO: 
- normal script with args and options
- more control on the ouput filenames and locations
- option for switching debug mode or not -- and changing the color ;)

Errors : 
- "à": ironically, most of the rare discourse instances are missed, because they take the incorrect form "A" in initial position (in est républicain)
- "vu" wrong when verb form
- "soit" wrong POS
- offsets shift on some files (mac encoding issue ?)


"""
import sys
import re
from collections import defaultdict
from AnnodisReader import *
from markers import LexConn
from datetime import datetime


# import argparse

# parser = argparse.ArgumentParser(description='preannotation of discourse markers lemma on annodis corpus')

# parser.add_argument('corpus',metavar='corpus dir',type=string ,help="")
# parser.add_argument('lex',metavar='lexicon',type=string,default=None ,help="")
# args = parser.parse_args()
# corpus = annodisCorpus(args.corpus,format="glozz")
# if args.lex is None:
#  lex = LexConn("../../../data/connecteurs/LexConn.prepNP.xml",version="2",stop=set())
# else: 
# lex = LexConn(args.lex,version="2",stop=set())


# todo as main 
if len(sys.argv)==2: 
    lex = LexConn("../../../data/connecteurs/LexConn.prepNP.xml",version="2",stop=set())
else:
    lex = LexConn(sys.argv[2],version="2",stop=set())

corpus = annodisCorpus(sys.argv[1],format="glozz")

template = """
<unit id="%s">
<metadata>
<author>%s</author>
<creation-date>%s</creation-date>
</metadata>
<characterisation>
<type>Connecteur</type>
<featureSet>
<feature name="cat">%s</feature>
<feature name="id">%s</feature>
<feature name="relation"></feature>
<feature name="relation_type">%s</feature>
</featureSet>
</characterisation>
<positioning>
<start>
<singlePosition index="%d"/>
</start>
<end>
<singlePosition index="%d"/>
</end>
</positioning></unit>
"""
AUTHOR = "automatic"
#DATE = str(datetime.today()).split(" ",1)[0]
DATE = str(0)
ID_PREFIX = "con"
DEBUG = True
RED = '\033[31m'
ENDC = '\033[0m'


all_forms = reduce(lambda x,y: x|y,[set(x.get_forms()) for x in lex])
# ad hoc processing of s' for si
all_forms.remove("s'")
all_forms.add("s'(?=ils?)")

# necessary to match longer markers first (eg "quand même" must be matched before "quand" and "même"
all_forms = list(all_forms)
all_forms.sort(key=len,reverse=True)

re_allconnectors = re.compile(ur"\b(%s)\b"%("|".join(all_forms)),flags=re.IGNORECASE | re.UNICODE)
res= defaultdict(list)
count = 1
for name,doc in corpus.docs().items():
    text = doc.text()
    match = re_allconnectors.finditer(text)
    for one in match:
        # position in text
        # lemme = type                              
        # todo: find the lemmatized version of text and check if correct (-> remove s'!)
        position = (one.start(),one.end())
        form = one.group(0).lower()
        res[name].append((form, position))
        marker = lex.get_by_form(form)
        #doc.add_connector(position,type=lex.get_by_form(form))
        # todo : refactor / method
        if len(marker)>1:
            print >> sys.stderr, "WARNING: ambiguous form, merging possible relations and id", form,  [x.id for x in marker]
        # also valid if only one marker    
        relations = reduce(lambda u,v:u|v,(set(x.relations) for x in marker))
        cid = ",".join((x.id for x in marker))
        cat = ",".join((x.cat for x in marker))
        new_element = template%(ID_PREFIX+str(count),AUTHOR,DATE,cat,cid,",".join(relations),position[0],position[1])
        xml = doc._doc._root
        xml.insert(-1,ET.fromstring(new_element))
        count = count + 1

    if DEBUG: # show the text with found markers in red
        print "="*10 + name + "="*10
        display = re.sub(re_allconnectors,RED+'\g<0>'+ENDC,text)
        print display.encode("utf8")

    
    #new_file = ".".join("".join(name.split("/")[-1:]).split(".")[:-1])+ "_conn.aa"
    new_file = "".join(name.split("/")[-1:])
    doc._doc.write(new_file,encoding="UTF-8",xml_declaration=True, method="xml")
    #doc.write(name=doc.basename()+"_conn.glozz",format="glozz")


