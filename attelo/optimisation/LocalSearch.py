#!/bin/env python
# -*- coding: iso-8859-1 -*-
#

"""
Classes wrapping various local search methods for optimisation

"""

import operator
import math
import random
import sys

class AbstractLocalSearch:
    """this is the common container for local search methods

    in theory just modify

    - constructor (not always)
    - acceptable
    - start
    - launch

    every search depends on an instance of a subclass of the {Problem} class
    """
    
    def __init__(self,instance):
        """creates instance of local search on problem instance
        """
        self.instance=instance
        self.currentSol=None
        self.currentVal=None
        self.bestSol=None
        self.neighbour=None
        self.bestVal=None

    # leave these alone
    def getCurrent(self):
	return self.currentSol

    def getBest(self):
	return self.bestSol

    def getNeighbour(self):
	return self.neighbour
    
    def getBestVal(self):
        return self.bestVal


    def acceptable(self,delta):
        """ abstract class for acceptability of move"""
        if self.instance.minimize():
            return (delta<0)
        else:
            return (delta>0)

    def move(self):
        # moves current solution and updates best solution
        self.currentSol=self.neighbour
	self.currentVal=self.instance.value(self.neighbour)
        if self.instance.minimize():
            comp=operator.lt
        else:
            comp=operator.gt
        if comp(self.currentVal,self.bestVal):
            self.improved=True
	    self.bestSol=self.currentSol
	    self.bestVal=self.currentVal
        else:
            self.improved=False

    def has_improved(self):
        return self.improved


    def delta(self):
        return self.instance.value(self.neighbour)-self.currentVal

    # default is random start
    def start(self):
        """ abstract class for starting local search with a solution"""
        self.bestSol=self.instance.randomStart()
        self.currentSol=self.bestSol
        self.bestVal=self.currentVal=self.instance.value(self.bestSol)

        
    def launch(self,maxIter,noBetterMax=10000):
        """abstract class for launching a local search with maxIter iterations

        
        """
        pass





class GreedySearch(AbstractLocalSearch):
    """greedy local search : random neighbour and move if better"""

    def start(self,mode="bf",threshold=0.0):
        """ abstract class for starting local search with a solution"""
        if mode=="random":
            self.bestSol=self.instance.randomStart()
        elif mode=="astar":
            self.bestSol=self.instance.specificStartAstar(threshold=threshold)
        elif mode=="beam":
            self.bestSol=self.instance.specificStartAstar(threshold=threshold,beam=100)
        else:
            self.bestSol=self.instance.specificStart()
        self.currentSol=self.bestSol
        self.bestVal=self.currentVal=self.instance.value(self.bestSol)

    def launch(self,maxIter,noBetterMax=10000,startmode="bf",threshold=0.0,verbose=False):
        """
        maxIter: max nb of iterations
        noBetterMax: max nb of iterations without without improving move
        """
        # iteration counter
        cpt=0
        # watch if no improving after many iterations
        noBetter=0
        self.start(mode=startmode,threshold=threshold)
        while (cpt<maxIter and noBetter<noBetterMax):
            self.neighbour=self.instance.randomNeighbour(self.currentSol)
            if self.acceptable(self.delta()):
                self.move()
                if verbose:
                    print >> sys.stderr, "improved:", self.currentVal
                noBetter=0
                #print self.currentVal
            else:
                noBetter+=1
            cpt+=1
        return cpt


class HillClimbingSearch(AbstractLocalSearch):
    """hill climbing local search : best neighbour and move always"""
    

    def __init__(self,instance):
        self.improved=False
        AbstractLocalSearch.__init__(self,instance)


    def start(self,mode="random",threshold=0.0):
        """ abstract class for starting local search with a solution"""
        if mode=="random":
            self.bestSol=self.instance.randomStart()
        elif mode=="astar":
            self.bestSol=self.instance.specificStartAstar(threshold=threshold)
        else:
            self.bestSol=self.instance.specificStart()
        self.currentSol=self.bestSol
        self.bestVal=self.currentVal=self.instance.value(self.bestSol)


    def acceptable(self,delta):
        return True

    def launch(self,maxIter,noBetterMax=100,startmode="bf",threshold=0.0,verbose=False):
        """
        maxIter: max nb of iterations
        noBetterMax: max nb of iterations without improving move
        """
        # iteration counter
        cpt=0
        # watch if no improving after many iterations
        noBetter=0
        self.start(mode=startmode,threshold=threshold)
        while (cpt<maxIter and noBetter<noBetterMax):
            self.neighbour=self.instance.bestNeighbour(self.currentSol)
            if self.neighbour is None:
                return cpt
            if self.acceptable(self.delta()):
                self.move()
                if self.has_improved():
                    noBetter=0
                    if verbose:
                        print >> sys.stderr, "improved:", self.currentVal
                else:
                    noBetter+=1
                #print self.currentVal
            else:
                noBetter+=1
            cpt+=1
        return cpt


class MultipleSearch:
    """local search with a number of random restarts

    needs a first solution + restart function + local search method (random or best)
    """
    pass


class SimulatedAnnealing(AbstractLocalSearch):
    """local search with simulated annealing"""
    

    def guessT0(self,threshold=0.80,sampleSize=100):
        """try to guess initial temperature
        
        must accept threshold percent of neighbours
        """
        self.start()
        sum_var=0.0
        for i in range(sampleSize):
            self.neighbour=self.instance.randomNeighbour(self.currentSol)
            sum_var+=self.delta()
        average=abs(sum_var/sampleSize)
        T0=-average/math.log(threshold)
        print >> sys.stderr, "init T0=",T0
        return T0

    def acceptable(self,delta):
        if self.instance.minimize():
            ok=(delta<0)
            factor=1
        else:
            ok=(delta>0)
            factor=-1
        if ok:
            return True
        else:
            T=float(self.T)
            proba=math.exp(-factor*delta/T)
            return (random.random()<proba)

    def launch(self,maxIter,T0,stageNb,stop="noimprovement",alpha=0.9):
        """
        T0 = initial temperature, if <0, method must make a guess
        stageNb: number of temperature stages
        stop: stopping method; defaut is "noimprovement", meaning
              stop when a stage does not improve the best solution found
        alpha= decreasing factor for T
        """
        stages=0
        noBetter=False
        totalct=0
        if T0<0:
            self.T=self.guessT0()
        else:
            self.T=T0
        
        self.start()
        while (stages<stageNb and not(noBetter)):
            cpt=0
            noBetter=True
            while (cpt<maxIter):
                self.neighbour=self.instance.randomNeighbour(self.currentSol)
                if self.acceptable(self.delta()):
                    self.move()
                    if self.has_improved():
                        noBetter=False
                cpt+=1
            self.T=alpha*self.T
            ##############
            stages+=1
            totalct+=cpt
        return totalct
  
if __name__=="__main__":
    import sys

    from Problem import TestProblem
    a=TestProblem()
    b=GreedySearch(a)
    iterNb=b.launch(10000000)
    print b.getBest(), b.getBestVal(),iterNb
    b=HillClimbingSearch(a)
    iterNb=b.launch(10000000)
    print b.getBest(), b.getBestVal(),iterNb

    b=SimulatedAnnealing(a)
    iterNb=b.launch(1000,-1,10)
    print b.getBest(), b.getBestVal(),iterNb

