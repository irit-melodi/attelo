#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" wrapper for GATE-like gazetteer

each item has a form, types and features
hier:type=dayspec:val=YESTERDAY

can be applied to add annotations to a text as 
a set of looked-up entities with offsets from the text

ex: 
text="Je vais tester ça lundi, ou alors demain."
lexicon.tag(text)
{
'date_key': [(35, 41, demain:subtype=None:val=FUTURE_REF)], 
'date': [(35, 41, demain:subtype=dayspec:val=TOMORROW)], 
'day': [(19, 24, lundi:subtype=None:val=1)]
}



TODO: 
- handle index file *.lst
"""
import sys
import os.path
import codecs
import glob
from collections import defaultdict
import re
import xml.etree.ElementTree as ET


class LookupItem:
    """
    class for mimicking gate gazetteer Lookup annotation
    """

    def __init__(self,major_type,description):
        """init with Lookup type and description according to gate syntax
        ex: 
        hier:type=dayspec:val=YESTERDAY
        """
        self._type=major_type
        parsed=description.split(":")
        self._form=parsed[0]

        feats=dict([x.split("=") for x in parsed[1:]])
        self._subtype=feats.get("type")
        self._val=feats.get("val")
        self._feats=feats
 

    def form(self):
        return self._form
   
    def majorType(self):
        return self._type

    def minorType(self):
        return self._subtype

    def val(self):
        return self._val

    def __repr__(self):
        return (u"%s:subtype=%s:val=%s"%(self.form(),self.minorType(),self.val())).encode("utf8")

    def __eq__(self,other):
        return self.form()==other.form()

    def export(self,target="LT-TTT"):
        """ export entry to target format"""
        if target=="LT-TTT":
            attrib=self._feats
            attrib["word"]=self.form()
            #attrib['case']="yes"
            res= ET.Element("lex",attrib=attrib)
            res.tail="\n"
            return res
        else:
            raise notImplementedError
        


class majorType:
    """ stores a set of entities of a given type. 
    can be used to search for given type in a string
    """

    def __init__(self,filename,encoding="utf8",comp=False):
        self._majorType=os.path.basename(filename).split(".")[0]
        self._items={}
        for line in codecs.open(filename,encoding=encoding):
            if line[0] not in ("%","/","#") and line.strip()!="":
                one=LookupItem(self._majorType,line.strip())
                self._items[one.form()]=one
        if comp:
            self.compile()
        else:
            self._regexp=None

    def compile(self):
        self._regexp=re.compile(r"|".join([r"(?:\b%s\b)"%x.replace(".","\.") for x in self._items.keys()]))
        #print >> sys.stderr, "|".join(["(?:%s)"%x for x in self._items.keys()])

    def __getitem__(self,key):
        return self._items.get(key)
    
    def items(self):
        return self._items

    def type(self):
        return self._majorType

    def tag(self,text):
        collect=[]
        if self._regexp is None:
            self.compile()
        else:
            for amatch in self._regexp.finditer(text):
                gotcha=(amatch.start(),amatch.end(),self[amatch.group(0)])
                if gotcha[-1] is not None:
                    collect.append(gotcha)
        return collect

    def export(self,target="LT-TTT",directory="."):
        """export to a lexicon file in target format"""
        if target=="LT-TTT":
            encoding="UTF-8"
            main=ET.Element("lexicon",attrib={'name':self.type()})
            filename=self._majorType+".xml"
        else:
            raise notImplementedError
        entries=[x.export(target=target) for x in self.items().values()]
        #python > 2.7 main.extend(entries)
        for x in entries:
            main.append(x)
        tree=ET.ElementTree(element=main)
        tree.write(directory+"/"+filename,encoding)
        # python2.7: tree.write(directory+"/"+filename,encoding=encoding,xml_declaration=True)

class Gazetteer:
    """ stores a bunch of entities of various types = gazetteer

    initialized from a directory containing a bunch of lexicons
    """
    def __init__(self,directory,encoding="utf8"):
        self._types={}
        if not(os.path.isdir(directory)):
            print >> sys.stderr, "directory does not exist"
            sys.exit(0)
        else:
            files=glob.glob(os.path.join(directory,"*.lst"))
            for one in files:
                try:
                    lexicon=majorType(one,encoding=encoding,comp=True)
                    self._types[lexicon.type()]=lexicon
                except:
                    print >> sys.stderr, "problem with", one
        

    def items(self):
        items=defaultdict(dict)
        for atype in self._types:
            for key in self._types[atype].items():
                items[key][atype]=self._types[atype][key]
        return items
    

    def __getitem__(self,key):
        all=[]
        for atype in self._types:
            value=self._types[atype][key]
            if value is not None:
                all.append(value)
        return all

    
    def lookup(self,key,type=None):
        if type is None:
            return self[key]
        else:
            return self._types[type][key]


    def tag(self,text):
        collect={}
        for atype in self._types:
            matched=self._types[atype].tag(text)
            if matched!=[]:
                collect[atype]=matched
        return collect

    def export(self,target="LT-TTT",directory="."):
        """export to a lexicon file in target format"""        
        for one in self._types:
            try:
                self._types[one].export(target=target,directory=directory)
                print >> sys.stderr, "done", one
            except:
                print >> sys.stderr, "encoding problem for %s ?"%one




if __name__=="__main__":
    
    testdir=sys.argv[1]
    lexicon=Gazetteer(testdir)

    lexicon.export(directory="./conversion")
    
    #all=lexicon.items()
    # gotit=lexicon.lookup(sys.argv[2])
    # for what in gotit:
    #     print what.majorType(), what

    # text="Je vais tester ça lundi, ou alors demain et mara."
    # result=lexicon.tag(text)
    # print result
    
