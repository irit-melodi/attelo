#!/bin/env python
# -*- coding: utf-8 -*-

"""
module for building discourse graphs from probability distribution and
respecting some constraints, using
Astar heuristics based search and variants (beam, b&b)

TODO: unlabelled evaluation seems to bug on RF decoding (relation is of type orange.value
-> go see in decoding.py)


"""
import random
import copy
import math
import sys
import numpy
from collections import defaultdict

from attelo.optimisation.Astar import State, Search, BeamSearch
from attelo.edu                import EDU

_class_schemes = {
    "subord_coord":{
        "subord":set(["elaboration", "e-elab", "attribution", "comment", "flashback", "explanation", "alternation"]),
        "coord": set(["continuation", "parallel", "contrast", "temploc", "frame", "narration", "conditional", "result", "goal", "background"]),
        "NONE":set(["null", "unknown", "NONE"])
        },
    # four class +/- closer to PDTB
    "pdtb": {
        "contingency": set(["explanation", "conditional", "result", "goal", "background"]),
        "temporal": set(["temploc", "narration", "flashback"]),
        "comparison":set(["parallel", "contrast"]),
        "expansion":set(["frame", "elaboration", "e-elab", "attribution", "continuation", "comment", "alternation"]),
        "error":set(["null", "unknown", "NONE"])
        },
    # our own hierarchy
    "minsdrt":{
        "structural": set(["parallel", "contrast", "alternation", "conditional"]),
        "sequence": set(["result", "narration", "continuation"]),
        "expansion": set(["frame", "elaboration", "e-elab", "attribution", "comment", "explanation"]),
        "temporal": set(["temploc", "goal", "flashback", "background"]),
        "error":set(["null", "unknown", "NONE"])
        }
}


subord_coord={
 "structural":"coord",
 "sequence":"coord",
 "expansion":"subord",
 "temporal":"subord",
 "error":"subord",
 }
     
for one in _class_schemes["minsdrt"]:
    for rel in _class_schemes["minsdrt"][one]:
        subord_coord[rel] = subord_coord[one]


class DiscData: 
    """ basic discourse data for a state: chosen links between edus at that stage + right-frontier state
    to save space, only new links are stored. the complete solution will be built with backpointers via the parent field
    """
    def __init__(self,parent=None,RF=[],tolink=[]):
        self._RF = RF
        self.parent = parent
        self._link = None
        self._tolink = tolink

    def accessible(self):
        return self._RF

    def final(self):
        return (self._tolink == [])

    def tobedone(self):
        return self._tolink

    def link(self,to_edu,from_edu,relation,RFC="full"):
        """
        RFC = "full": use the distinction coord/subord
        RFC = "simple": consider everything as subord   
        RFC = "none" no constraint on attachment
        """
        if to_edu not in self.accessible():
            print >> sys.stderr, "error: unreachable node", to_edu, "(ignored)"
        else:
            index = self.accessible().index(to_edu)
            self._link = (to_edu,from_edu,relation)
            # update the right frontier -- coord relations replace their attachment points, subord are appended, and evrything below disappear from the RF
            # unknown relations are subord
            #print >> sys.stderr, type(relation)
            #print >> sys.stderr, map(type,subord_coord.values())
            if RFC=="full" and subord_coord.get(relation,"subord")=="coord":
                self._RF = self._RF[:index]
            elif RFC=="simple":
                self._RF = self._RF[:index+1]
            else: # no RFC, leave everything. 
                pass
            self._RF.append(from_edu)

    def __str__(self):
        return str(self._link)+"/ RF="+str(self._RF)+"/ to attach = "+" ".join(map(str,self._tolink))

    def __repr__(self):
        return str(self)

class DiscourseState(State):
    """
    instance of discourse graph with probability for each attachement+relation on a subset
    of edges.

    implements the State interface to be used by Search

    strategy: at each step of exploration choose a relation between two edus
    related by probability distribution

    'data' is set of instantiated relations (typically nothing at the beginning, but could be started with a few chosen relations)
    'shared' points to shared data between states (here proba distribution between considered pairs of edus at least, but also can include precomputed info for heuristics)
    

    
    """
    def __init__(self,data,heuristics,shared):
        self._data=data
        self._cost=0
        self._shared=shared
        self._h=heuristics(self)
        
        
    def data(self):
        return self._data

    def proba(self,edu_pair):
        return self._shared["probs"].get(edu_pair,("no",None))

    def shared(self):
        return self._shared

    def strategy(self):
        """ full or not, if the RFC is applied to labelled edu pairs
        """
        return self._shared["RFC"]
    
    # solution found when everything has been instantiated
    # TODO: adapt to disc parse, according to choice made for data
    def isSolution(self):
        return self.data().final()

 
    def nextStates(self):
        """must return a state and a cost
        TODO: adapt to disc parse, according to choice made for data -> especially update to RFC
        """
        all=[]
        one=self.data().tobedone()[0]
        #print ">> taking care of node ", one
        for attachmt in self.data().accessible():
            # FIXME: this might is problematic because we use things like
            # checking if an EDU is in some list, which fails on object id
            new = copy.deepcopy(self.data())
            new.tobedone().pop(0)
            relation,pr = self.proba((attachmt,one))
            if pr is not None:
                new.link(attachmt,one,relation,RFC=self.strategy())
                new.parent = self.data()
                if self._shared["use_prob"]:
                    if pr==0:
                        score = -numpy.inf
                    else:
                        score = -math.log(pr)
                    all.append((new,score))
                else:
                    all.append((new,pr))
        return all

    def __str__(self):
        return str(self.data())+": "+str(self.cost())

    
    def __repr__(self):
        return str(self.data())+": "+str(self.cost())


    # heuristiques
    def h_average(self):
        # return the average probability possible when n nodes still need to be attached
        # assuming the best overall prob in the distrib
        missing_links = self.data().tobedone()
        #try:
        if self.shared()["use_prob"]:
            transform = lambda x: -math.log(x) if x!=0 else -numpy.inf
        else:
            transform = lambda x:x
        try:
            pr = sum(map(transform,[self.shared()["heuristics"]["average"][x] for x in missing_links]))
        except:
            print >> sys.stderr, missing_links
            print >> sys.stderr, self.shared()["heuristics"]["average"][x] 
            sys.exit(0)
        return pr

    def h_best_overall(self):
        # return the best probability possible when n nodes still need to be attached
        # assuming the best overall prob in the distrib
        missing_links = len(self.data().tobedone())
        pr = self.shared()["heuristics"]["best_overall"]
        if self.shared()["use_prob"]:
            score = -math.log(pr) if pr!=0 else -numpy.inf
            return (score*missing_links)
        else:
            return (pr*missing_links)


    def h_best(self):
        # return the best probability possible when n nodes still need to be attached
        # assuming the best overall prob in the distrib
        missing_links = self.data().tobedone()
        if self.shared()["use_prob"]:
            transform = lambda x: -math.log(x) if x!=0 else -numpy.inf
        else:
            transform = lambda x:x
        pr = sum(map(transform,[self.shared()["heuristics"]["best_attach"][x] for x in missing_links]))
        return pr





class DiscourseSearch(Search):

    def newState(self,data):
        return DiscourseState(data,self._hFunc,self.shared())

    def recover_solution(self,endstate):
        # follow back pointers to collect list of chosen relations on edus. 
        all=[]
        current = endstate.data()
        while current.parent is not None:
            #print current
            all.append(current._link)
            current = current.parent
        all.reverse()
        return all



class DiscourseBeamSearch(BeamSearch):

    def newState(self,data):
        return DiscourseState(data,self._hFunc,self.shared())

    def recover_solution(self,endstate):
        # follow back pointers to collect list of chosen relations on edus. 
        all=[]
        current = endstate.data()
        while current.parent is not None:
            #print current
            all.append(current._link)
            current = current.parent
        all.reverse()
        return all

  
h0=(lambda x: 0.)
h_max=DiscourseState.h_best_overall
h_best=DiscourseState.h_best
h_average= DiscourseState.h_average

def preprocess_heuristics(prob_distrib):
    """precompute a set of useful information used by heuristics, such as
             - best probability 
             - table of best probability when attaching a node, indexed on that node
             
    format of prob_distrib is format given in main decoder: a list of (arg1,arg2,proba,best_relation)
    """
    result = {}
    result["best_overall"] = max([x[2] for x in prob_distrib])

    result["best_attach"]=defaultdict(float)
    result["average"]=defaultdict(list)
    for (a1,a2,p,r) in prob_distrib:
        result["best_attach"][a2.id] = max(result["best_attach"][a2.id],p)
        result["average"][a2.id].append(p)

    for one in result["average"]:
        result["average"][one] = sum(result["average"][one])/len(result["average"][one])
    #print >> sys.stderr, result
    return result

def prob_distrib_convert(prob_distrib):
    """convert a probability distribution table to desired input for a* decoder
    NOT IMPLEMENTED: to be factored in from astar_decoder
    """
    pass
    




# TODO: order function should be a method parameter
# - root should be specified ? or a fake root ? for now, it is the first edu
# - should allow for (at least local) argument inversion (eg background), for more expressivity
def astar_decoder(prob_distrib,heuristics=h0,beam=None,RFC="simple",use_prob=True,nbest=1,**kwargs):
    """wrapper for astar decoder to be used by processing pipeline

    - heuristics is a* heuristic funtion (estimate the cost of what has not been explored yet)
    - prob_distrib is a list of (a1,a2,p,r): scores p on each possible edge (a1,a2), and the best label r corresponding to that score 
    - use_prob: indicates if previous scores are probabilities in [0,1] (to be mapped to -log) or arbitrary scores (untouched)
    - beam: size of the beam-search (if None: vanilla astar)
    - RFC: whether to use a "simple" right-frontier-constraint, ~= every relation is subord, or the "full" RFC (falls back to simple
    in case of unlabelled predictions)
    TODO: nbest=n generates the best n solutions. done at search level as a generator, but propagatingwould break interface with prediction
    solution: store the n_best somewhere ...  
    """
    prob = {}
    edus = set()
    for (a1,a2,p,r) in prob_distrib:
        #print r
        prob[(a1.id,a2.id)]=(r,p)
        edus.add((a1.id,int(a1.start)))
        edus.add((a2.id,int(a2.start)))

    edus = list(edus)
    edus.sort(key = lambda x: x[1])
    edus = map(lambda x:x[0],edus)
    print >> sys.stderr, "\t %s nodes to attach"%(len(edus)-1)

    pre_heurist = preprocess_heuristics(prob_distrib)
    if beam:
        a = DiscourseBeamSearch(heuristic=heuristics,shared={"probs":prob,"use_prob":use_prob,"heuristics":pre_heurist,"RFC":RFC},queue_size=beam)
    else:
        a = DiscourseSearch(heuristic=heuristics,shared={"probs":prob,"use_prob":use_prob,"heuristics":pre_heurist,"RFC":RFC})
        
    genall = a.launch(DiscData(RF=[edus[0]],tolink=edus[1:]),norepeat=True,verbose=False)
    endstate = genall.next()
    sol =  a.recover_solution(endstate)
    return sol


if __name__=="__main__":
    import sys
    import time
    from pprint import pprint

    def mkFakeEDU(id):
        return EDU(id, 0, 0, "x");

    edus = map(mkFakeEDU, ["x0","x1","x2","x3","x4"])

    # would result of prob models  max_relation (p(attachement)*p(relation|attachmt))  
    prob_distrib=[
        (edus[1],edus[2],0.6,'elaboration'),
        (edus[2],edus[3],0.3,'narration'),
        (edus[1],edus[3],0.4,'continuation'),
        ]
    for one in edus[1:-1]:
        prob_distrib.append((one,edus[4],0.1,'continuation'))


    pre_heurist = preprocess_heuristics(prob_distrib)

    prob = {}
    for (a1,a2,p,r) in prob_distrib:
        prob[(a1,a2)]=(r,p)

    


    t0=time.time()
    a = DiscourseSearch(heuristic=h_average,shared={"probs":prob,"heuristics":pre_heurist,"use_prob":True,"RFC":"full"})
    genall = a.launch(DiscData(RF=[edus[1]],tolink=edus[2:]),norepeat=True,verbose=True)
    endstate = genall.next()
    print "total time:",time.time()-t0
    sol =  a.recover_solution(endstate)
    print "solution:", sol
    print "cost:", endstate.cost()
    print a.iterations
    print "total time:", time.time()-t0
    # a = DiscourseSearch(heuristic=h2, shared={"probs":prob,"nodes":edus[2:]})
    # sol=a.launch(DiscData(nodes=edus[1:2],RF=edus[1:2]),norepeat=True)
    # print sol
    # print sol.cost()
    # print a.iterations
    # print "total time:",time.time()-t0

    # test with discourse input file (.features)
    print "new function test"
    
    print astar_decoder(prob_distrib)
