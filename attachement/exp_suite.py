#!/usr/bin/env python

import os
from expsuite import PyExperimentSuite
from experiment import Experiment


class MySuite(PyExperimentSuite):

    def reset(self, params, rep):
        train_dir = params['train_dir']
        test_dir = params['test_dir']
        model_type = params["model"]
        update = params["update"]
        avg = params["avg"]
        C = params["aggressiveness"] # only for PA
        self._exp = Experiment(train_dir,
                               test_dir,
                               model_type=model_type,
                               update=update,
                               avg=avg,
                               C=C)        
        # settings for training
        self._epochs = params["epochs"]
        return
        
    
    def iterate(self, params, rep, n):        
        self._exp.train( self._epochs )
        res = self._exp.test()
        return res




if __name__ == "__main__": 
    mysuite = MySuite() 
    mysuite.start() 
