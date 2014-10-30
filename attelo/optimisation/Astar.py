#!/bin/env python
# -*- coding: utf-8 -*-
#

"""
Various search algorithms for combinatorial problems:

 ok - Astar (shortest path with heuristics)
    - variants:
      ok   - beam search (astar with size limit on waiting queue)
      ok   - nbest solutions: implies storing solutions and a counter, and changing return values
             (actually most search will make use of a recover_solution(s) to reconstruct desired data)
         - branch and bound (astar with forward lookahead)
  
"""

import sys
import heapq
from pprint import pformat

class State:
    """
    state for state space exploration with search
    
    
    (contains at least state info and cost)
    
    
    """
    def __init__(self,data,heuristics):
        self._data = data
        self._cost = 0
        self._h = heuristics(data)
        
    def cost(self):
        return self._cost

    def data(self):
        return self._data

    def updateCost(self,value):
        self._cost +=value
        
    def __eq__(self,other):
        return self.data() == other.data()

    def __lt__(self,other):
        f1 = self.cost()+self._h
        f2 = other.cost()+other._h
        if f1<f2:
            return True
        elif f1==f2:
            return self.cost()>other.cost()
        else:
            return False

    def __hash__(self):
        return hash(self.data())

    # must be defined by new class
    #def isSolution(self):
    #    return False
    # must be overloaded
    #def nextStates(self):
    #    return []



class Search:
    """abstract class for search
    each state to be explored must have methods
       - nextStates() (successor states + costs)
       - isSolution() (is the state a valid solution)
       - cost()       (cost of the state so far)
       cost must be additive
       
    default is astar search (search the minimum cost from init state to a solution

    heuristic: heuristics guiding the search (applies to state-specific data(), see State)
    shared: other data shared by all nodes (eg. for heuristic computation ?)

    queue_size: limited beam-size to store states. (commented out, pending tests). 
    
    """
    
    def __init__(self,heuristic = (lambda x: 0.),shared = None,queue_size = None):
        self._todo = []
        self._seen = {}
        self._hFunc = heuristic
        self._shared = shared
        self._queue_size = queue_size

    def resetQueue(self):
        self._todo = []
        
    def resetSeen(self):
        self._seen = {}


    def shared(self):
        return self._shared

    # change this to change the method
    def newState(self,data):
        return State(data,self._hFunc)

    def addQueue(self,items,ancestorCost):
        # each item must be a successor and a cost
        for one,cost in items:
            s = self.newState(one)
            s.updateCost(ancestorCost+cost)
            heapq.heappush(self._todo,s)
        #if self._queue_size is not None:
        #    new  =  heapq.nsmallest(self._queue_size,self._todo)
        #    self._todo  =  new

    def getBest(self):
        return heapq.min(self._todo)

    def popBest(self):
        return heapq.heappop(self._todo)
        
    def emptyQueue(self):
        return (self._todo ==[])

    def alreadySeen(self,e):
        return hash(e) in self._seen

    def addSeen(self,e):
        self._seen[hash(e)]=e

    def launch(self,initState,verbose=False,norepeat=False):
        """launch search from initital state value

        norepeat means there's no need for an "already seen states" datastructure
        
        Todo: should be able to change the queue_size here 
        """
        self.resetQueue()
        if not(norepeat):
            self.resetSeen()
        self.addQueue([(initState,0)],0.)
        self.iterations=0
        
        while not(self.emptyQueue()):
            skip = False
            self.iterations += 1
            e=self.popBest()
            if verbose:
                print "\033[91mcurrent best state", pformat(e.__dict__),  "\033[0m"
                print 'states todo=',self._todo
                print 'seen=',self._seen
            if not(norepeat):
                if self.alreadySeen(e):
                    skip = True
                    if verbose:
                        print 'already seen', e
                else:
                    skip = False
            if not(skip):
                if e.isSolution():
                    yield e
                else:
                    if not(norepeat):
                        self.addSeen(e)
                    next=e.nextStates()
                    #print next
                    self.addQueue(next,e.cost())
            if verbose:
                        print 'update:'
                        print 'states todo=', self._todo
                        print 'seen=', self._seen

        # if it comes to that, there is no solution
        raise StopIteration




class BeamSearch(Search):
    """
    search with heuristics but limited size waiting queue
    (restrict to p-best solutions at each iteration)
    """
    def __init__(self,heuristic=(lambda x: 0.),shared=None,queue_size=10):
        self._todo=[]
        self._seen={}
        self._hFunc=heuristic
        self._queue_size=queue_size
        self._shared=shared
    
    def addQueue(self,items,ancestorCost):
        # each item must be a successor and a cost
        for one,cost in items:
            s=self.newState(one)
            s.updateCost(ancestorCost+cost)
            heapq.heappush(self._todo,s)
        if self._queue_size is not None:
            new=heapq.nsmallest(self._queue_size,self._todo)
            self._todo=new

class TestState(State):
    """dummy cost uniform search: starting from int, find minimum numbers
    of step to get to 21, with steps being either +1 or *2 (and +2 to get faster to sol)

    data = (current value,operator list)  

    from start 0, must return 6 (+1 *2 *2 +1 *2 *2 +1)
    """
    _ops = {"*2":(lambda x:2*x),
            "+1":(lambda x:x+1),
            "-1":(lambda x:x-1),
            "+2":(lambda x:x+2),
            }

    def _update_data(self,op):
        return (self._ops[op](self.data()[0]),self.data()[1]+op)

    def isSolution(self):
        return ((self.data()[0])==21)

 
    def nextStates(self):
        return [(self._update_data(x),1.) for x in self._ops]

    def __str__(self):
        return str(self.data())+":"+str(self.cost())

    
    def __repr__(self):
        return str(self.data())+":"+str(self.cost())



class TestSearch(Search):

    def newState(self,data):
        return TestState(data,self._hFunc)

class TestSearch2(BeamSearch):

    def newState(self,data):
        return TestState(data,self._hFunc)



if __name__=="__main__":
    from math import log
    #init test state, defined here as a value and a string storing operators
    a=(0,"")
    from math import log
    # dumb testing heuristics assuming we can *2 to victory. log base 2.3 is to prevent wrong limit conditions
    # and force h to be optimistic
    h_bete = lambda x: (log(abs(21-x[0]),2.3) if x[0]!=21 else 0)
    h0 = lambda x: 0
    for (name,b) in (("UC",TestSearch(h0)),
                     ("Astar",TestSearch(h_bete)),
                     ("Beam/h/100 1-",TestSearch2(h_bete,queue_size=100)),
                     ("Beam/h/100 2-",TestSearch(h_bete,queue_size=100)),
                     ("Beam/h0/100",TestSearch2(h0,queue_size=100))):
        gc=b.launch(a,verbose=False)
        tot = 5
        nbest = tot
        print "============testing %d-best for %s"%(nbest,name)
        while nbest>0:
            c = gc.next()
            print "solution no %d"%(tot+1-nbest)
            if c is None:
                print "--- no solution found"
                break
            else:
                print "solution %d =  ?"%c.cost(), c
            print "explored states =", len(b._seen)
            nbest = nbest - 1

        
