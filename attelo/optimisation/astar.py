#!/bin/env python
# -*- coding: utf-8 -*-
#

"""
Various search algorithms for combinatorial problems:

* [OK] Astar (shortest path with heuristics),
  and variants:

  * [OK] beam search (astar with size limit on waiting queue)
  * [OK] nbest solutions: implies storing solutions and a counter, and changing
         return values (actually most search will make use of a
         recover_solution(s) to reconstruct desired data)
  * branch and bound (astar with forward lookahead)
"""

from __future__ import print_function
import sys
import heapq
from pprint import pformat

class State:
    """
    state for state space exploration with search


    (contains at least state info and cost)


    """
    def __init__(self, data, heuristics):
        self._data = data
        self._cost = 0
        self._h = heuristics(data)

    def cost(self):
        return self._cost

    def data(self):
        return self._data

    def updateCost(self, value):
        self._cost += value

    def __eq__(self, other):
        return self.data() == other.data()

    def __lt__(self, other):
        f1 = self.cost() + self._h
        f2 = other.cost() + other._h
        if f1 < f2:
            return True
        elif f1 == f2:
            return self.cost() > other.cost()
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

    * :py:meth:`nextStates` - successor states + costs
    * :py:meth:`isSolution` - is the state a valid solution
    * :py:meth:`cost` - cost of the state so far (must be additive)

    default is astar search (search the minimum cost from init state to a solution

    :param heuristic: heuristics guiding the search (applies to state-specific
                      data(), see :py:class:`State`)

    :param shared: other data shared by all nodes (eg. for heuristic computation)

    :param queue_size: limited beam-size to store states. (commented out,
                       pending tests)
    """
    def __init__(self,
                 heuristic=lambda x: 0.,
                 shared=None,
                 queue_size=None):
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
    def newState(self, data):
        return State(data, self._hFunc)

    def addQueue(self, items, ancestorCost):
        # each item must be a successor and a cost
        for one, cost in items:
            s = self.newState(one)
            s.updateCost(ancestorCost+cost)
            heapq.heappush(self._todo, s)
        #if self._queue_size is not None:
        #    new = heapq.nsmallest(self._queue_size, self._todo)
        #    self._todo = new

    def getBest(self):
        return heapq.min(self._todo)

    def popBest(self):
        return heapq.heappop(self._todo)

    def emptyQueue(self):
        return self._todo == []

    def alreadySeen(self, e):
        return hash(e) in self._seen

    def addSeen(self, e):
        self._seen[hash(e)] = e

    def launch(self, initState,
               verbose=False,
               norepeat=False):
        """launch search from initital state value

        norepeat means there's no need for an "already seen states" datastructure

        Todo: should be able to change the queue_size here
        """
        self.resetQueue()
        if not norepeat:
            self.resetSeen()
        self.addQueue([(initState, 0)], 0.)
        self.iterations = 0

        while not self.emptyQueue():
            skip = False
            self.iterations += 1
            e = self.popBest()
            if verbose:
                print("\033[91mcurrent best state", pformat(e.__dict__), "\033[0m")
                print('states todo=', self._todo)
                print('seen=', self._seen)
            if not norepeat:
                if self.alreadySeen(e):
                    skip = True
                    if verbose:
                        print('already seen', e)
                else:
                    skip = False
            if not skip:
                if e.isSolution():
                    yield e
                else:
                    if not norepeat:
                        self.addSeen(e)
                    next = e.nextStates()
                    #print(next)
                    self.addQueue(next, e.cost())
            if verbose:
                print('update:')
                print('states todo=', self._todo)
                print('seen=', self._seen)

        # if it comes to that,  there is no solution
        raise StopIteration




class BeamSearch(Search):
    """
    search with heuristics but limited size waiting queue
    (restrict to p-best solutions at each iteration)
    """
    def __init__(self,
                 heuristic=lambda x: 0.,
                 shared=None,
                 queue_size=10):
        self._todo = []
        self._seen = {}
        self._hFunc = heuristic
        self._queue_size = queue_size
        self._shared = shared

    def addQueue(self, items, ancestorCost):
        # each item must be a successor and a cost
        for one, cost in items:
            s = self.newState(one)
            s.updateCost(ancestorCost+cost)
            heapq.heappush(self._todo, s)
        if self._queue_size is not None:
            new = heapq.nsmallest(self._queue_size, self._todo)
            self._todo = new
