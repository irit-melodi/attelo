#!/usr/bin/env python
"""
processing discourse experiment combining model of 
attachment + model of relation conditioned on attachment
and evaluation 

various methods can be called for training, decoding and evaluation
(decoding is externally defined)

"""
import sys
import collections
import orange, orngTest, orngStat


    



class Experiment:

    def __init__(self,train_file,test_file):
        """discourse experiment: train and test 
        are sets of instances with 
        meta data for instances = file id + instance id 

        experiment is made of training model, decoding method and evaluation method

        attachement only so far
        """
        self._train = orange.ExampleTable(train_file)
        self._test =orange.ExampleTable(test_file)



    def train_model(self,filename=None,method="NB"):
        """load a model from file
        method can be a few baseline method or a function on the saved model
        """
        if filename is not None: 
            self._model = method(file)
        else:
            if method=="NB":
                bayes = orange.BayesLearner(self._train,adjustThreshold=True) 
                bayes.name="naive bayes"
                self._model = bayes
            else:
                print >> sys.stderr, "method not callable", method
                sys.exit(0)


    def add_conditional_labelling(self,train_file):
        """include model for relation given attachment
        TODO
        """
        pass

    def best_label(self,instance):
        """returns best label for given instance, with proba. 
        """
        distrib = self._label_model(instance,orange.GetProbabilities)[1]
        distrib.sort(reverse=True)
        best_p,best_answer = distrib[0] 
        return best_p,best_answer.getclass()

    def decoding(self,method="local",sorting=lambda x:x):
        """call decoding on each test file, producing results for instances
        of test files
        default is local evaluation of classifier on each instance independently
        sorting orders decision within groups (eg. id of firstnode,id of secondnode)
        """
        by_files = collections.defaultdict(list)
        for one in self._test: 
            probs = self._model(one, orange.GetProbabilities)[1]
            #probs.sort()
            #probs.reverse()
            #p,answer = probs[0] 
            #is_correct = (answer.getclass()=="True")            
            by_files[one["m#FILE"]].append(((one["m#FirstNode"],one["m#SecondNode"]),probs))

        score = 0
        self._result = {}
        if method="local":
            base = len(self._test)
            self._base = base
        else:
            self._base = len(by_files)
        for afile in by_files:
            if method in ["astar","last"]:
                # TODO: include EDU sorting condition
                decisions = sorted(by_files[afile],key=sorting)
            else:
                decisions = by_files[afile]
            self._result[afile] = decode_one(decisions,method=method)

                
    def evaluate(self,method="unlabelled",grouping=None):
        score = 0 
        base = self._base
        for onefile in self._result:
            decisions = self._result[onefile]
            for onedecision in decisions:
                if self.reference(onefile,onedecision,which="attachment")==decisions[onedecision]:
                    score += 1
        return (score,base)
                



if __name__="__main__":
    exp = Experiment(sys.argv[1],sys.argv[2])
    exp.train_model(method="NB")
    exp.decoding(method="local")
    exp.evaluate(method="unlabelled")
