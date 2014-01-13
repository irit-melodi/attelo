#!/bin/env python
# -*- coding: utf-8 -*-
#
""" interface for problem subject to local optimisation (used in {Search})

subclass it with you own ...
"""


class Problem:


    # those needs to be overloaded
    def __init__(self,*args):
        pass

    def loadInstance(self,filename):
        pass
    
    def randomStart(self):
        pass

    def specificStart(self):
        pass

    def randomNeighbour(self,current):
        pass

    def bestNeighbour(self,current):
        pass

    # defaut is to try to minimize
    def minimize(self):
        return True

    def value(self,x):
        pass

import random
class TestProblem(Problem):
    """basic single variable function for testing purposes"""

    def __init__(self):
        self.func=lambda x: abs(4.0*x*x-64)
        
    def minimize(self):
        return True
    
    def value(self,x):
        return self.func(x)

    def randomStart(self):
        return random.random()*20

    def specificStart(self):
        return 0.0

    def randomNeighbour(self,current):
        return current+0.1*(-1)**(random.randint(1,2))

    def bestNeighbour(self,current):
        a,b=current+0.1,current-0.1
        if self.value(a)<=self.value(b):
            return a
        else:
            return b
        

  
if __name__=="__main__":
    import sys

    a=TestProblem()
    print a.randomStart()
    print a.bestNeighbour(10)
    print a.specificStart()
    print a.randomNeighbour(10)
