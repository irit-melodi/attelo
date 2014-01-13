#!/usr/bin/env python

import os
import sys

from polka.classification.classifier import PerceptronClassifier, PAClassifier, print_results as cl_pprint
from polka.ranking.ranker import PerceptronRanker, PARanker, print_results as rk_pprint
from polka.common.datasource import ClassificationSource, RankingSource
from polka_conversion import PolkaConvertor
import tempfile
import numpy


_DEBUG=False


class Experiment:

    def __init__(self, train_dir, test_dir, features2cross=[], features2filter=[],
                 model_type="ranker", update="pa", avg=False, C=1000000):
        self._train_dir = train_dir
        self._test_dir = test_dir
        self._feats2cross = features2cross
        self._feats2filter = features2filter
        self._model_type = model_type
        self._update = update
        self._avg = avg
        self._C = C
        self._model = self.select_model()
        return


    def select_model(self):
        if self._model_type == "classifier":
            if self._update == "perc":
                model = PerceptronClassifier( avg=self._avg )
            elif self._update == "pa":
                model = PAClassifier( avg=self._avg, C=self._C )
        elif self._model_type == "ranker":
            if self._update == "perc":
                model = PerceptronRanker( avg=self._avg )
            elif self._update == "pa":
                model = PARanker( avg=self._avg, C=self._C )
        return model


    def convert_data(self, data, format="ranker"):
        conv = PolkaConvertor( data,
                               features2cross=self._feats2cross,
                               features2filter=self._feats2filter)
        _,polkafile = tempfile.mkstemp()
        conv.dump_rank_instances(polkafile)
        if format == "classifier":
            conv.dump_class_instances(polkafile)
        elif format == "ranker":
            conv.dump_rank_instances(polkafile)
        return polkafile


    def train(self, epochs):
        train_file = self.convert_data(self._train_dir,
                                       format=self._model_type)
        self._model.learn( train_file, epochs=epochs )
        os.unlink(train_file)
        return


    def test(self):
        test_file = self.convert_data(self._test_dir,
                                      format=self._model_type)
        res = self._model.evaluate( test_file )
        os.unlink(test_file)
        if self._model_type == "classifier":
            cl_pprint( res )
        elif self._model_type == "ranker":
            rk_pprint( res )
        return res


    def test_and_analyze(self, features=[]):
        correct = 0.0
        total = 0.0
        feat2correct = {}
        feat2total = {}
        model = self._model
        # ranker
        if self._model_type == "ranker":
            test_file = self.convert_data(self._test_dir,
                                          format="ranker")
            src = RankingSource( test_file )
            src.set_alphabet( model.get_alphabet() )
            for inst in src:
                # print inst
                gold_indices = inst.get_gold_indices()
                pred_index, _, _ = model.rank( inst )
                pred_cand_feat_dict = dict(inst.get_candidate_inputs()[pred_index][1])
                total += 1
                if _DEBUG:
                    print >> sys.stderr, "%s: Pred index: %s <==> Gold indices: %s" \
                          %(inst._index,pred_index,gold_indices)
                for f in features:
                    # print f, pred_cand_feat_dict.get(f)
                    val = pred_cand_feat_dict.get( f, None )
                    if val != None:
                        feat2total[(f,val)] = feat2total.get( (f,val) , 0 ) + 1 
                if pred_index in gold_indices:
                    correct += 1
                    for f in features:
                        # if f in pred_cand_feat_dict:
                        val = pred_cand_feat_dict.get( f, None )
                        if val != None:
                            feat2correct[(f,val)] = feat2correct.get( (f,val) , 0 ) + 1             
        # classifier
        if self._model_type == "classifier":
            test_file = self.convert_data(self._test_dir,
                                          format="classifier")
            src = ClassificationSource( test_file )
            src.set_alphabets( model.get_alphabets() )
            for inst in src:
                label, _ = model.classify( inst )
                pred_cand_feat_dict = dict(inst.get_input())
                total += 1
                for f in features:
                    # print f, pred_cand_feat_dict
                    #if f in pred_cand_feat_dict:
                    #    feat2total[f] = feat2total.get(f,0) + 1
                    val = pred_cand_feat_dict.get( f, None )
                    if val != None:
                        feat2total[(f,val)] = feat2total.get( (f,val) , 0 ) + 1 
                if label == inst.get_target_label():
                    correct += 1
                    for f in features:
                        #if f in pred_cand_feat_dict:
                        #    feat2correct[f] = feat2correct.get(f,0) + 1
                        val = pred_cand_feat_dict.get( f, None )
                        if val != None:
                            feat2correct[(f,val)] = feat2correct.get( (f,val) , 0 ) + 1  
        
        res = dict(correct=correct, total=total,
                   correct_per_feat=feat2correct,
                   total_per_feat=feat2total)

        # print res
        acc = correct / float( total )
        print "Overall ACC: %s (%s/%s)" %(round(acc,3),correct,total)
        print "\nAccuracy wrt specific features:"
        print "%-40s %10s %10s %10s" %("Feature", "Correct", "Total", "Acc.")
        for fv in feat2total.keys():
            correct1 = feat2correct.get(fv,0)
            total1 = feat2total.get(fv,0)
            acc1 = 0.0
            if total1 > 0:
                acc1 = correct1 / float( total1 )
            print "%-40s %10s %10s %10s" %(str(fv)[:40],correct1,total1,round(acc1,3))
        
        return res
        

if __name__ == "__main__":

    from optparse import OptionParser

    # options
    parser = OptionParser()
    parser.add_option("-m", "--model_type",
                      choices=['classifier', 'ranker'],
                      default='ranker',
                      help="Type of model: 'classifier', 'ranker' (def.)",
                      metavar="MODEL")
    parser.add_option("-u", "--update",
                      choices=["perc", "pa"], 
                      default="pa",
                      help="Learning update: perc(eptron), pa(ssive-aggressive) (def.)", 
                      metavar="UP")
    parser.add_option("-a", "--averaging",
                      default=False,
                      action="store_true",
                      help="Parameter averaging", 
                      metavar="AVG")
    parser.add_option("-C", "--C", \
                      action="store", \
                      default=numpy.inf, \
                      type=float, \
                      help="aggressiveness parameter for PA (default: inf)")
    parser.add_option("-d", "--train",
                      action="store",
                      default='',
                      help="training data directory",
                      metavar="TRAIN")
    parser.add_option("-i", "--iterations",
                      action="store",
                      default=1,
                      type=int,
                      help="Number of epochs",
                      metavar="ITER")
    parser.add_option("-f", "--features",
                      action="store",
                      default="D#DIRECTIONALITY_LEFT=True,D#N_EDU=True,D#LAST=True,C#PAR_DIST",
                      type=str,
                      help="Features used for error analysis",
                      metavar="FEATS")
    parser.add_option("-k", "--crossing",
                      default="",
                      action="store",
                      help="features to cross with others (only binary for now: e.g., D#DIRECTIONALITY_LEFT) (default: '')")
    parser.add_option("-l", "--filter",
                      default=None,
                      action="store", 
                      help="comma-separated list of features to be filtered")

   
    
    # parser.add_option("-c", "--cutoff", # TODO
    #                   action="store",
    #                   default=1,
    #                   type=int,
    #                   help="frequency cutoff for features",
    #                   metavar="CUTOFF")
    (options, args) = parser.parse_args()

    test_dir = args[0]    
    train_dir = options.train
    feats2cross = []
    feats2filter = []
    if options.crossing:
        feats2cross = options.crossing.split(",")
    feats4err_analysis = []
    if options.features:
        feats4err_analysis = options.features.split(",") 
    if options.filter:
        feats2filter = options.filter.split(",")
        
    exp = Experiment(train_dir,
                     test_dir,
                     model_type=options.model_type,
                     features2cross=feats2cross,
                     features2filter=feats2filter,
                     update=options.update,
                     avg=options.averaging,
                     C=options.C)
    exp.train( options.iterations )
    # exp.test()
    exp.test_and_analyze( features=feats4err_analysis )


    #feats.filter_feats(feature_list,mode="lose",continuous=True)
