#!/bin/env python
# -*- coding: iso-8859-1 -*-
#

"""
Various search algorithms for combinatorial problems:

 ok - Astar (shortest path with heuristics)
 - variants:
    ok   - beam search (astar with size limit on waiting queue)
         - branch and bound (astar with forward lookahead)
         - nbest solutions: implies storing solutions and a counter, and changing return values
         (actually most search will make use of a recover_solution(s) to reconstruct desired data)

"""

import heapq
from pprint import pformat

class State:
    """
    state for state space exploration with search
    
    
    (contains at least state info and cost)
    
    
    """
    def __init__(self,data,heuristics):
        self._data=data
        self._cost=0
        self._h=heuristics(data)
        
    def cost(self):
        return self._cost

    def data(self):
        return self._data

    def updateCost(self,value):
        self._cost +=value
        
    def __eq__(self,other):
        return self.data()==other.data()

    def __lt__(self,other):
        f1=self.cost()+self._h
        f2=other.cost()+other._h
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

    heuristic: heuristics guiding the search
    shared: data shared by all nodes (eg. for heuristic computation ?)
    """
    
    def __init__(self,heuristic=(lambda x: 0.),shared=None):
        self._todo=[]
        self._seen={}
        self._hFunc=heuristic
        self._shared=shared

    def resetQueue(self):
        self._todo=[]
        
    def resetSeen(self):
        self._seen={}


    def shared(self):
        return self._shared

    # change this to change the method
    def newState(self,data):
        return State(data,self._hFunc)

    def addQueue(self,items,ancestorCost):
        # each item must be a successor and a cost
        for one,cost in items:
            s=self.newState(one)
            s.updateCost(ancestorCost+cost)
            heapq.heappush(self._todo,s)

    def getBest(self):
        return heapq.min(self._todo)

    def popBest(self):
        return heapq.heappop(self._todo)
        
    def emptyQueue(self):
        return (self._todo==[])

    def alreadySeen(self,e):
        return hash(e) in self._seen

    def addSeen(self,e):
        self._seen[hash(e)]=e

    def launch(self,initState,verbose=False,norepeat=False):
        """launch search from initital state value

        norepeat means there's no need for an "already seen states" datastructure
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
        new=heapq.nsmallest(self._queue_size,self._todo)
        self._todo=new

class TestState(State):
    """dummy cost uniform search: starting from int, find minimum numbers
    of step to get to 21, with steps being either +1 or *2

    from start 0, must return 7 (+1 *2 *2 +1 *2 *2 +1)
    """

    def isSolution(self):
        return ((self.data())==21)

 
    def nextStates(self):
        return [(self.data()*2,1.),(self.data()+1,1.),(self.data()+2,1.)]

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
    a=0
    # 
    b1 = TestSearch((lambda x: 0))
    gc1 = b1.launch(a,verbose=True)
    c1 = gc1.next()
    print "testing uniform cost"
    if c1 is None:
        print "--- no solution found"
    else:
        print "--- solution %d = 6 ?"%c1.cost()
    print "--- explored states =", len(b1._seen)
    b2=TestSearch((lambda x: 0))
    c2=b2.launch(a,verbose=False).next()
    print "testing Astar"
    if c2 is None:
        print "--- no solution found"
    else:
        print "solution %d = 6 ?"%c2.cost()
    print "explored states =", len(b2._seen)
    b3=TestSearch()
    c3=b3.launch(a,verbose=False).next()
    print "testing Astar"
    if c3 is None:
        print "--- no solution found"
    else:
        print "solution %d = 6 ?"%c3.cost()
    print "explored states =", len(b3._seen)
    #b3=TestSearch2((lambda x: 0),queue_size=10)
    b3=TestSearch((lambda x: 0))
    gc3=b3.launch(a,verbose=False)
    nbest = 2
    while nbest>0:
        c3 = gc3.next()
        print "testing 2-best for Astar"
        if c3 is None:
            print "--- no solution found"
            break
        else:
            print "solution %d = 6 ou 7 ?"%c3.cost()
        print "explored states =", len(b3._seen)
        nbest = nbest - 1
        
