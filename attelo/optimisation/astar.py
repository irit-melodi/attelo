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
import heapq
from pprint import pformat

# pylint: disable=too-few-public-methods

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
        "past path cost"
        return self._cost

    def data(self):
        "actual distinguishing contents of a state"
        return self._data

    def updateCost(self, value):
        self._cost += value

    def __eq__(self, other):
        return self.data() == other.data()

    def __lt__(self, other):
        f_self = self.cost() + self._h
        f_other = other.cost() + other._h
        if f_self < f_other:
            return True
        elif f_self == f_other:
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
        """
        Add a set of succesors to the search queue

        :type items [(data, float)]
        """
        # each item must be a successor and a cost
        for one, cost in items:
            succ = self.newState(one)
            succ.updateCost(ancestorCost+cost)
            heapq.heappush(self._todo, succ)
        #if self._queue_size is not None:
        #    new = heapq.nsmallest(self._queue_size, self._todo)
        #    self._todo = new

    def popBest(self):
        """
        Return and remove the lowest cost item from the search queue
        """
        return heapq.heappop(self._todo)

    def emptyQueue(self):
        """
        Return `True` if the search queue is empty
        """
        return self._todo == []

    def alreadySeen(self, state):
        """
        Return `True` if the given search state has already been seen
        """
        return hash(state) in self._seen

    def addSeen(self, state):
        """
        Mark a state as seen
        """
        self._seen[hash(state)] = state

    def launch(self, initState,
               verbose=False,
               norepeat=False):
        """launch search from initital state value

        :param: norepeat: there's no need for an "already seen states" datastructure
        """
        # TODO: should be able to change the queue_size here
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
                    nxt = e.nextStates()
                    #print(next)
                    self.addQueue(nxt, e.cost())
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
            succ = self.newState(one)
            succ.updateCost(ancestorCost+cost)
            heapq.heappush(self._todo, succ)
        if self._queue_size is not None:
            new = heapq.nsmallest(self._queue_size, self._todo)
            self._todo = new
