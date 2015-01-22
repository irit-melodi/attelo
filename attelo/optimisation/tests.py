"""
attelo.optimisation tests
"""

from __future__ import print_function
from math import log
import sys
import unittest

from .astar import BeamSearch, Search, State

# pylint: disable=too-few-public-methods, protected-access

class TestState(State):
    """dummy cost uniform search: starting from int, find minimum numbers of
    step to get to 21, with steps being either +1 or *2 (and +2 to get faster
    to sol)

    data = (current value,operator list)

    from start 0, must return 6 (+1 *2 *2 +1 *2 *2 +1)
    """
    _ops = {"*2": lambda x: 2*x,
            "+1": lambda x: x+1,
            "-1": lambda x: x-1,
            "+2": lambda x: x+2}

    def __init__(self, data, heuristic):
        super(TestState, self).__init__(data, future_cost=heuristic(data))

    def _update_data(self, opr):
        '''return the updated value and operator list we'd get from
        applying `opr` to the current value'''
        left = self.data()[0]
        right = self.data()[1]
        return (self._ops[opr](left), right+opr)

    def is_solution(self):
        return (self.data()[0]) == 21

    def next_states(self):
        return [(self._update_data(x), 1.) for x in self._ops]

    def __str__(self):
        return str(self.data())+":"+str(self.cost())

    def __repr__(self):
        return str(self.data())+":"+str(self.cost())


class TestSearch(Search):
    'Test instance of the A* search algorithm'
    def new_state(self, data):
        return TestState(data, self._h_func)

class TestBeamSearch(BeamSearch):
    'Test instance of the A* search algorithm (beam variant)'
    def new_state(self, data):
        return TestState(data, self._h_func)


class AstarTest(unittest.TestCase):
    'Tests for the core of the A* algorithm'

    def _test_search(self, name, search, tot=5):
        '''
        Get the `tot` best solutions
        '''
        gen = search.launch((0, ""), verbose=False)
        nbest = tot
        print("============testing %d-best for %s" % (nbest, name),
              file=sys.stderr)
        while nbest > 0:
            soln = gen.next()
            print("solution no %d"%(tot+1-nbest),
                  file=sys.stderr)
            if soln is None:
                print("--- no solution found",
                      file=sys.stderr)
                break
            else:
                print("solution %d =  ?" % soln.cost(), soln,
                      file=sys.stderr)
                print("explored states =", len(search._seen),
                      file=sys.stderr)
            nbest = nbest - 1


    def test_astar(self):
        '''
        init test state, defined here as a value and a string storing operators
        dumb testing heuristics assuming we can *2 to victory. log base 2.3
        is to prevent wrong limit conditions
        and force h to be optimistic
        '''
        h_bete = lambda x: log(abs(21-x[0]), 2.3) if x[0] != 21 else 0
        h_zero = lambda x: 0
        tests = [("UC", TestSearch(h_zero)),
                 ("Astar", TestSearch(h_bete)),
                 ("Beam/h/100 1-", TestBeamSearch(h_bete, queue_size=100)),
                 ("Beam/h/100 2-", TestSearch(h_bete, queue_size=100)),
                 ("Beam/h0/100", TestBeamSearch(h_zero, queue_size=100))]
        for name, search in tests:
            self._test_search(name, search)
