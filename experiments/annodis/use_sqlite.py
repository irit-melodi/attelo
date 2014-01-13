#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""test sqlite3 pour accès voisins

une fois la base indexée sur un lemme (le premier par exemple), explose pytables les doigts dans le nez

"""

import sqlite3 
import sys
import time
from pprint import pformat
from collections import defaultdict
import operator


_fields=["lemme1","cat1","rel1","lemme2","cat2","rel2","jacc","lin"]
#def get_voisins_list(lemma_list,table,seuil=None,chunk=30,isHashed=False)
#def init_voisins(filename="voisins.db"):


def make_db(filename,dest_name,mkindex=True):
    data=open(filename)
    # throw out header
    data.next()
    # create db
    con = sqlite3.connect(dest_name)
    con.execute("create table voisins(lemme1,cat1,rel1,lemme2,cat2,rel2,jacc,lin)")
    for one in data:
        one=one.decode("latin").strip().split("\t")
        one[-1]=float(one[-1])
        # should be buffered for efficiency
        con.executemany("insert into voisins(lemme1,cat1,rel1,lemme2,cat2,rel2,jacc,lin) values (?,?,?,?,?,?,?,?)",(one,))
    data.close()
    if mkindex:
        con.execute("create index idxlemma1 on voisins(lemme1)")
    con.close()


def test_db(dbfile,list_lemme=["pomme de terre"]):
    t0=time.time()
    res=[]
    con = sqlite3.connect(dbfile)
    for one_lemme in list_lemme:
        for row in con.execute("select lemme1,lemme2,lin from voisins where lemme1=?",(one_lemme,)):
            res.append(row)
    print time.time()-t0,"s"
    return res


def init_voisins(filename):
    con = sqlite3.connect(filename)
    return con,con

from copy import copy

# some arguments are just for compatibility with h5 version
def get_voisins_list(lemma_list,table,seuil=None,isHashed=False):
    req="select lemme1,lemme2,lin,cat1,cat2,rel1,rel2 from voisins where lemme1=?"
    if seuil is not None:
        req="select lemme1,lemme2,lin,cat1,cat2,rel1,rel2 from voisins where lemme1=?"+" and lin>%f"%seuil
    else:
        pass
    
    res=[]
    for one_lemme in lemma_list:
        #print >> sys.stderr, req, (one_lemme,)
        for row in table.execute(req,(one_lemme,)):
            res.append(row)
    res=[Voidis(**dict(zip(["lemme1","lemme2","lin","cat1","cat2","rel1","rel2"],x))) for x in res]
    return res

def get_voisins_dict(lemma_list,table,seuil=None):
    """get voisins from list of lemma in table, with threshold=seuil,
    put them in a dict, separating different functions

    TODO: 
       - options to collapse similar lemmas with diff functions
       - index on lemmas, functions, or total
    """
    lv = get_voisins_list(table,lemma_list,seuil=seuil)
    voisins=dict_from_table(lv)
    return voisins

def dict_from_table(voisins_list,filter_set=None,nbest=-1):
      """
      a partir du resultat d'une requete, index les voisins par items

      si filter_set est présent ne garde que les éléments lexicaux présents
      """
      res=defaultdict(list)
      for one in voisins_list:
            w1,w2,lin=one.lemme1,one.lemme2,one.lin
            if filter_set is not None and w2 in filter_set:
                  res[w1].append((w2,lin))
            else:
                  res[w1].append((w2,lin))
      if nbest!=-1:
            for w1 in res:
                  keep=sorted([(y,x) for (x,y) in res[w1]])
                  keep.reverse()
                  keep=keep[:nbest]
                  res[w1]=[(y,x) for (x,y) in keep]
      return res


def relation_match(rel1,rel2):
    """are the two syntactic relations comparable ?"""
    return (rel1==rel2 or (rel1=="obj" and rel2=="de")) 


def collapse_vsn(vsnList,cumul=operator.__add__):
    """collapse voisins d'un item ayant le même lemme+categorie,
    et des relations comparables (ex suj/suj)
    
    """
    vsnDict=defaultdict(list)
    source=vsnList[0].lemme1
    for one in vsnList:
        vsnDict[one.lemme2].append(one)

    res={}
    for one in vsnDict:
        total=0
        vsn=vsnDict[one]
        for x in vsn:
            if relation_match(x.rel1,x.rel2):
                total = cumul(total,x.lin)
        res[one]=total
    return res



#TODO class item lexical ...
#TODO croisement avec le lefff ?
# passer lefff sous sqlite ?
# -> class Item pour lefff a fusionner ici

class Voidis:
    """class for pairs of lexical neighbours

    TODO: fonctions
       - ajout info mutuelle
       - ajout appartenance synonymes dicosyn
       - appartenance des lemmes au lexique dicosyn
       - frequence des lemmes ?
    """
    
    def __init__(self,**kwargs):
        self.__dict__=kwargs

    def __str__(self):
        return pformat(self.__dict__)

    def __repr__(self):
        return pformat(self.__dict__)






if __name__=="__main__":
   
    if False:
        make_db(sys.argv[1],"../data/voisins.db")
    liste_test=sys.argv[1:]
    t0=time.time()
    filename,table=init_voisins("../data/voisins.db")
    r=get_voisins_list(liste_test,table)
    print >> sys.stderr, "done in %f s"%(time.time()-t0)
    print [x.lemme2 for x in r]
