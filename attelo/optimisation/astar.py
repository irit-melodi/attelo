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
from abc import ABCMeta, abstractmethod
from six import with_metaclass
import heapq
from pprint import pformat

# pylint: disable=too-few-public-methods
# pylint: disable=abstract-class-not-used, abstract-class-little-used


class State(with_metaclass(ABCMeta, object)):
    """
    state for state space exploration with search

    (contains at least state info and cost)

    Note the functions `is_solution` and `next_states` which
    must be implemented
    """
    def __init__(self, data, cost=0, future_cost=0):
        self._data = data
        self._cost = cost
        self._future_cost = future_cost

    def cost(self):
        "past path cost"
        return self._cost

    def future_cost(self):
        "future cost"
        return self._future_cost

    def total_cost(self):
        "past and future cost"
        return self._cost + self._future_cost

    def data(self):
        "actual distinguishing contents of a state"
        return self._data

    def update_cost(self, value):
        "add to the current cost"
        self._cost += value

    def __eq__(self, other):
        return self.data() == other.data()

    def __lt__(self, other):
        f_self = self.total_cost()
        f_other = other.total_cost()
        if f_self < f_other:
            return True
        elif f_self == f_other:
            return self.cost() > other.cost()
        else:
            return False

    def __hash__(self):
        return hash(self.data())

    @abstractmethod
    def is_solution(self):
        "return `True` if the state is a valid solution"
        raise NotImplementedError

    @abstractmethod
    def next_states(self):
        "return the successor states and their costs"
        raise NotImplementedError


class Search(with_metaclass(ABCMeta, object)):
    """abstract class for search
    each state to be explored must have methods

    * :py:meth:`next_states` - successor states + costs
    * :py:meth:`is_solution` - is the state a valid solution
    * :py:meth:`cost` - cost of the state so far (must be additive)

    default is astar search (search the minimum cost from init state to a
    solution

    :param heuristic: heuristics guiding the search (applies to state-specific
                      data(), see :py:class:`State`)

    :param shared: other data shared by all nodes (eg. for heuristic
                   computation)

    :param queue_size: limited beam-size to store states. (commented out,
                       pending tests)
    """
    def __init__(self,
                 heuristic=lambda x: 0.,
                 shared=None,
                 queue_size=None):
        self._todo = []
        self._seen = {}
        self._h_func = heuristic
        self._shared = shared
        self._queue_size = queue_size
        self.iterations = 0

    def reset_queue(self):
        "Clear out the search queue"
        self._todo = []

    def reset_seen(self):
        "Mark every state as not yet seen"
        self._seen = {}

    def shared(self):
        "Information that can be shared across states"
        return self._shared

    @abstractmethod
    def new_state(self, data):
        "Build a new state from the given data"
        raise NotImplementedError

    def add_queue(self, items, ancestor_cost):
        """
        Add a set of succesors to the search queue

        :type items [(data, float)]
        """
        # each item must be a successor and a cost
        for one, cost in items:
            succ = self.new_state(one)
            succ.update_cost(ancestor_cost+cost)
            heapq.heappush(self._todo, succ)
        # if self._queue_size is not None:
        #    new = heapq.nsmallest(self._queue_size, self._todo)
        #    self._todo = new

    def pop_best(self):
        """
        Return and remove the lowest cost item from the search queue
        """
        return heapq.heappop(self._todo)

    def has_empty_queue(self):
        """
        Return `True` if the search queue is empty
        """
        return self._todo == []

    def is_already_seen(self, state):
        """
        Return `True` if the given search state has already been seen
        """
        return hash(state) in self._seen

    def add_seen(self, state):
        """
        Mark a state as seen
        """
        self._seen[hash(state)] = state

    def launch(self, init_state,
               verbose=False,
               norepeat=False):
        """launch search from initital state value

        :param: norepeat: there's no need for an "already seen states"
                          datastructure
        """
        # TODO: should be able to change the queue_size here
        self.reset_queue()
        if not norepeat:
            self.reset_seen()
        self.add_queue([(init_state, 0)], 0.)
        self.iterations = 0

        while not self.has_empty_queue():
            skip = False
            self.iterations += 1
            state = self.pop_best()
            if verbose:
                print("\033[91mcurrent best state", pformat(state.__dict__),
                      "\033[0m")
                print('states todo=', self._todo)
                print('seen=', self._seen)
            if not norepeat:
                if self.is_already_seen(state):
                    skip = True
                    if verbose:
                        print('already seen', state)
                else:
                    skip = False
            if not skip:
                if state.is_solution():
                    yield state
                else:
                    if not norepeat:
                        self.add_seen(state)
                    nxt = state.next_states()
                    # print(next)
                    self.add_queue(nxt, state.cost())
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
        super(BeamSearch, self).__init__(heuristic=heuristic,
                                         shared=shared,
                                         queue_size=queue_size)

    def new_state(self, data):
        raise NotImplementedError

    def add_queue(self, items, ancestor_cost):
        # each item must be a successor and a cost
        for one, cost in items:
            succ = self.new_state(one)
            succ.update_cost(ancestor_cost + cost)
            heapq.heappush(self._todo, succ)
        if self._queue_size is not None:
            new = heapq.nsmallest(self._queue_size, self._todo)
            self._todo = new
