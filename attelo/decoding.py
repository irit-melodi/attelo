"""
July 2012

attachment decoding: from local prediction, optimize a discourse structure while respecting
a chosen set of constraints, such as: MST decoding, incremental decoding with a right frontier constraint, etc

should regroup
      x- MST decoding
      x- A* decoding with various heuristics
      x- baseline: local, small beam search

example: cf coling_expes.sh



TODO:
 - GET RID OF THE FCKG GLOBAL VARIABLES
        - meta-feature names that are used for indexing/... etc. they mess things up in "online_learner" too
         FILE, edu ids, and span ids. 
        - class for parser ? would help ! but first put cfg everywhere
 x- might be useful to have project config files for that instead of option switch ... 
 - abstract main as processing method, depending on various things: fold nb, learner, decoder, eval function for one discourse
and within that, abstract layers : fold,document
 - more generic descriptions of features names
 - n best out of A* decoding, to prepare ranking
                x- generate all solution in generic astar
                 - add parameter to decoding+process_document ? but the whole API for prediction assumes only one prediction
                 - revamp scores to allow eval in nbest list (eg: at given depth and/or oracle in list; what else?)
 - other evals: tree-edit, parseval-like, other?
 x- RFC with coord-subord distinction
 - nicer report for scores (table, latex, figures)
""" 
import os
import sys
import cPickle
from ConfigParser import ConfigParser
import Orange
try:
    STATS=True
    from scipy.stats import wilcoxon, ttest_rel, mannwhitneyu, sem, bayes_mvs
except:
    print >> sys.stderr, "no module scipy.stats, cannot test stat. significance of results"
    STATS=False

from fileNfold import make_n_fold, makeFoldByFileIndex
from ParseSearch import astar_decoder, h0, h_best, h_max, h_average
from attachment.mst import MST_list_edges as MST_decoder
from attachment.greedy import locallyGreedy, getSortedEDUs
from megam_wrapper import MaxentLearner
from online_learner import Perceptron, StructuredPerceptron
from edu import EDU

# from MST import MSTdecoder


# index names for EDU pairs 
# FirstNode = "SOURCE"
# SecondNode = "TARGET"
# TargetSpanStart = "TargetSpanStart"
# TargetSpanEnd = "TargetSpanEnd"
# SourceSpanStart = "SourceSpanStart"
# SourceSpanEnd = "SourceSpanEnd"
# FILE = "FILE"
annodis_cfg = {
            "FirstNode" : "SOURCE",
            "SecondNode" : "TARGET",
            "TargetSpanStart" : "TargetSpanStart",
            "TargetSpanEnd" : "TargetSpanEnd",
            "SourceSpanStart" : "SourceSpanStart",
            "SourceSpanEnd" : "SourceSpanEnd",
            "FILE" : "FILE",
            "CLASS": "CLASS"
            }

stac_cfg = {
        "FirstNode"       : "id_DU1",
        "SecondNode"      : "id_DU2",
        "TargetSpanStart" : "start_DU2",
        "TargetSpanEnd"   : "end_DU2",
        "SourceSpanStart" : "start_DU1",
        "SourceSpanEnd"   : "end_DU1",
        "FILE"            : "dialogue",
        "CLASS"           : "CLASS" }

def_cfg = annodis_cfg

def local_baseline(prob_distrib, threshold = 0.5, use_prob=True):
    """just attach locally if prob is > threshold
    """
    predicted = []
    for (arg1, arg2, probs, label) in prob_distrib:
        attach = probs
        if use_prob:
            if attach > threshold:
                predicted.append((arg1.id, arg2.id, label))
        else:
            if attach >= 0.0:
                predicted.append((arg1.id, arg2.id, label))
    return predicted


def last_baseline(prob_distrib, use_prob=True):
    "attach to last, always"
    edus = getSortedEDUs(prob_distrib)
    ordered_pairs = zip(edus[:-1],edus[1:])
    dict_prob = {}
    for (a1,a2,p,r) in prob_distrib:
        dict_prob[(a1.id,a2.id)]=(r,p)

    predicted=[(a1.id,a2.id,dict_prob[(a1.id,a2.id)][0]) for (a1,a2) in ordered_pairs]
    return predicted
              
        
    




def discourse_eval(predicted, data, labels = None, debug = False, cfg = def_cfg ):
    """basic eval: counting correct predicted edges (labelled or not)
    data contains the reference attachments
    labels the corresponding relations
    cfg: stores index names of important meta-features (edu ids, file id, etc)
    """
    #print "REF:", data
    #print "PRED:", predicted
    score = 0
    dict_predicted = dict([((a1, a2), rel) for (a1, a2, rel) in predicted])
    for one in data:
        arg1 = one[cfg["FirstNode"]].value
        arg2 = one[cfg["SecondNode"]].value
        if debug:
            print >> sys.stderr, arg1, arg2, dict_predicted.get((arg1, arg2))
        if dict_predicted.has_key((arg1, arg2)):
            if labels is None:
                score += 1
                if debug: print >> sys.stderr, "correct"
            else:
                relation_ref = labels.filter_ref({cfg["FirstNode"]:[arg1], cfg["SecondNode"]:[arg2]})
                if len(relation_ref) == 0:
                    print >> sys.stderr, "attached pair without corresponding relation", one[cfg["FILE"]], arg1, arg2
                else:
                    relation_ref = relation_ref[0][cfg["CLASS"]].value
                    score += (dict_predicted[(arg1, arg2)] == relation_ref)
    #print "SCORE:", score
    return score

def combine_probs(attach_instances, relation_instances, attachmt_model, relations_model, cfg = def_cfg):
    """retrieve probability of the best relation on an edu pair, given the probability of an attachment
    """
    # !! instances set must correspond to same edu pair in the same order !!
    distrib = []
    rel = relation_instances.domain[cfg["CLASS"]]
    attach_instances = sorted(attach_instances, key = lambda x:x.get_metas())
    relation_instances = sorted(relation_instances, key = lambda x:x.get_metas())

    for (i, one) in enumerate(attach_instances):
        p_attach = attachmt_model(one, Orange.classification.Classifier.GetProbabilities)[1]
        p_relations = relations_model(relation_instances[i], Orange.classification.Classifier.GetBoth)
        if not(instance_check(one, relation_instances[i],cfg=cfg)): print >> sys.stderr, "mismatch of attacht/relation instance, instance number", i, meta_info(one), meta_info(relation_instances[i])
        # this should be investigated
        try: best_rel = p_relations[0].value
        except: best_rel = p_relations[0]
        
        rel_prob = max(p_relations[1])
        distrib.append((EDU(one[arg1].value, one[SourceSpanStartIndex].value, one[SourceSpanEndIndex].value, one[FILEIndex].value),
                         EDU(one[arg2].value, one[TargetSpanStartIndex].value, one[TargetSpanEndIndex].value, one[FILEIndex].value),
                         p_attach * rel_prob,
                         best_rel))
    return distrib


def index_by_metas(instances,metas=None):
    """transform a data table to a dictionary of instances indexed by ordered tuple of all meta-attributes; 
    convenient to find instances associated to multiple tables (eg edu pairs for attachment+relations)
    """
    if metas is None:
        to_keep = lambda x: x.get_metas().values()
    else:
        to_keep = lambda x: [x[y] for y in metas] 
    result = [(tuple([y.value for y in to_keep(x)]),x) for x in instances]
    return dict(result)
    


def add_labels(predicted, rel_instances, relations_model, cfg = def_cfg ):
    """ predict labels for a given set of edges (=post-labelling an unlabelled decoding)
    """
    rels = index_by_metas(rel_instances,metas=[cfg["FirstNode"],cfg["SecondNode"]])
    result = []
    for (a1,a2,_r) in predicted:
        instance_rel = rels[(a1,a2)]
        rel = relations_model(instance_rel,Orange.classification.Classifier.GetValue)
        result.append((a1,a2,rel))
    return result

def instance_check(one, two, cfg = def_cfg ):
    return (one[cfg["FirstNode"]] == two[cfg["FirstNode"]]) and (one[cfg["SecondNode"]] == two[cfg["SecondNode"]]) and (one[cfg["FILE"]] == two[cfg["FILE"]])

def meta_info(instance):
    return "%s: %s-%s" % (instance[cfg["FILE"]], instance[cfg["FirstNode"]], instance[cfg["SecondNode"]])


def exportGraph(predicted, doc, folder):
    fname = os.path.join(folder, doc + ".rel")
    if not os.path.exists(folder):
        os.makedirs(folder)
    f = open(fname, 'w')
    for (a1, a2, rel) in predicted:
        f.write(rel + " ( " + a1 + " / " + a2 + " )\n")
    f.close()



def process_document(onedoc, model, decoder, data_attach,
                     with_relations = False, data_relations = [], model_relations = None,
                     save_results = False, output_folder = None,
                     threshold = None,
                     unlabelled = False,
                     post_labelling=False,
                     use_prob=True,
                     cfg = def_cfg):
    """decode one document (onedoc), selecting instances for attachment from data_attach, (idem relations if present),
    using trained model,model

    TODO: check that call to learner can be uniform with 2 parameters (as logistic), as the documentation is inconsistent on this
    """
    
   
    FILE = cfg["FILE"]
    CLASS = cfg["CLASS"]
    # TODO: should be added to config at the start
    TargetSpanStartIndex = data_attach.domain.index(metacfg["TargetSpanStart"])
    TargetSpanEndIndex = data_attach.domain.index(metacfg["TargetSpanEnd"])
    SourceSpanStartIndex = data_attach.domain.index(metacfg["SourceSpanStart"])
    SourceSpanEndIndex = data_attach.domain.index(metacfg["SourceSpanEnd"])
    FILEIndex = data_attach.domain.index(metacfg["FILE"])

    doc_instances = data_attach.filter_ref({FILE : onedoc})

    if with_relations:
        rel_instances = data_relations.filter_ref({FILE : onedoc})
    if with_relations and not(post_labelling):
        prob_distrib = combine_probs(doc_instances, rel_instances, model, model_relations, cfg = cfg)
    else:
        # home-made online models
        if model.name in ["Perceptron", "StructuredPerceptron"]:
            prob_distrib = model.get_scores( doc_instances, use_prob=use_prob )
        # orange-based models
        else:
            prob_distrib = [(EDU(one[arg1].value, one[SourceSpanStartIndex].value, one[SourceSpanEndIndex].value, one[FILEIndex].value),
                             EDU(one[arg2].value, one[TargetSpanStartIndex].value, one[TargetSpanEndIndex].value, one[FILEIndex].value),
                             model(one, Orange.classification.Classifier.GetProbabilities)[1],
                             "unlabelled") for one in doc_instances]
    # print prob_distrib
        
    # get prediction (input is just prob_distrib)
    if threshold:
        predicted = decoder(prob_distrib, threshold = threshold, use_prob=use_prob)
        # predicted = decoder(prob_distrib, threshold = threshold)
    else:
        predicted = decoder(prob_distrib, use_prob=use_prob)
        # predicted = decoder(prob_distrib)
        
    if post_labelling:
        predicted = add_labels(predicted, rel_instances, model_relations, use_prob=use_prob, cfg = cfg)
        # predicted = add_labels(predicted, rel_instances, model_relations)

    # print predicted

    # prediction scoring
    if save_results:
        exportGraph(predicted, onedoc, output_folder)
    # eval for that prediction
    doc_ref = doc_instances.filter_ref({CLASS : "True"})
    if with_relations and not(unlabelled):
        labels = rel_instances.filter_ref({CLASS:["UNRELATED"]}, negate = 1)
    else:
        labels = None
    #print "REF:", doc_ref
    #print "PRED:", predicted
    one_score = structure_eval(predicted, doc_ref, labels = labels)
    total_ref = len(doc_ref)
    total_pred = len(predicted)
    return one_score, total_pred, total_ref

def f1_score(p_r):
    (prec,recall) = p_r
    return 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0

class Report:
    """class holding experiments results, for nice output (not implemented)

    todo: ? correction built-in the saved scores or saved ? 
    """
    def __init__(self, evals, params = None, correction = 1.0):
        correct, total_pred, total_ref = map(lambda x:sum(x), zip(*evals))
        all_correct, all_pred, all_total = zip(*evals)
        self._measures = {}
        self._measures["prec"] = zip(all_correct,all_pred)
        self._measures["recall"] = zip(all_correct,all_total)
        # behold the power of functional programming !
        prec_all = map(lambda (x,y): x/float(y) if y> 0 else 0,self._measures["prec"])
        rec_all = map(lambda (x,y): x/float(y) if y> 0 else 0,self._measures["recall"])
        self._measures["F1"] = map(f1_score,zip(prec_all,rec_all))

        prec = float(correct) / total_pred if total_pred > 0 else 0
        recall = float(correct) / total_ref if total_ref > 0 else 0
        self.f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0
        self.prec = prec
        self.recall = recall
        self.correct = correct
        self.total_pred = total_pred
        self.total_ref = total_ref
        self.correction = correction
        self.params = params
        if self.correction < 1.0:
            corr_r = self.recall * self.correction
            self.f1_corr = (2 * prec * corr_r / (prec + corr_r)) if (prec + corr_r) > 0 else 0
            self.corr_r = corr_r
        else:
            self.f1_corr = None
            self.corr_r = None

    def table(self, mode = "latex"):
        # plain, latex
        pass

    def save(self,name):
        dname = os.path.dirname(name)
        if not os.path.exists(dname):
            os.makedirs(dname)
        reportfile=open(name,"w")
        cPickle.dump(self,reportfile)
        reportfile.close()


    def plot(self, mode = "roc"):
        # plot curve:  learning curve, nbest solutions, what else ?
        pass

    def summary(self):
        output = ["{relations} {context} ({relnb}) : \t {decoders}+{learners}, h={heuristics}, unlabelled={unlabelled},post={post_label},rfc={rfc}".format(**self.params.__dict__)]
        prec, recall, f1 = self.prec, self.recall, self.f1
        try:
            m,(a,b) = self.confidence_interval()
        except:
            print >> sys.stderr, "warning: not able to compute confidence interval"
            m, a = (0,0)
        output.append( "\t Prec=%1.3f, Recall=%1.3f, F1=%1.3f +/- %1.3f (%1.3f +- %1.3f)" % (prec, recall, f1,self.standard_error(),m,(m-a)))
        if self.f1_corr:
            output.append( "\t with recall correction estimate, R=%1.3f, F1=%1.3f" % (self.corr_r, self.f1_corr))
        return " ".join(output)

    def standard_error(self,measure="F1"):
        """standard error (of the mean) on measures by text (TODO: folds should be saved too ?)
        """
        return sem(self._measures[measure])

    def confidence_interval(self,measure="F1",alpha=0.95):
        # will return mean, ci 
        return bayes_mvs(self._measures[measure],alpha)[0]


    # inspired by Apetite Evaluation class
    def significance(self,other,measure="F1",test="wilcoxon"):
        """computes stats significance of difference between two sets of scores
        test can be paired wilcoxon, mannwhitney for indep samples, or paired ttest. 
        """
        d1 = self._measures[measure]
        d2 = other._measures[measure]
        if type(d1[0])==type(1.0) or type(d1[0])==type(1):
            pass
        else:
            d1 = [x for (x,y) in d1]
            d2 = [x for (x,y) in d2]
          
        differences = [(x,y) for (x,y) in zip(d1,d2) if x!=y] 
        #print >> sys.stderr, differences
        #print >> sys.stderr, d2
        #print >> sys.stderr, [x for (i,x) in enumerate(d1) if x!=d2[i]]
        assert len(d1)==len(d2)           
        
        results = {}
            
        if test=="wilcoxon" or test == "all":
            tscore,p = wilcoxon(d1,d2)
            results["wilcoxon"] = p
        if test == "ttest" or  test == "all":
            tscore,p = ttest_rel(d1,d2)
            results["paired ttest"] = p
        if test =="mannwhitney" or  test == "all":
            tscore,p = mannwhitneyu(d1,d2)
            results["mannwhitney"] = p 
        return results



#######################


# --first results 10- fold
# windowed(5)/train/naive bayes
# - local: Prec=0.619, Recall=0.638, F1=0.628
# -RFC beam(2):  Prec=0.610, Recall=0.676, F1=0.641
# - MST Prec=0.671, Recall=0.744, F1=0.706
# - RFC h_best: 
# - RFC h_moyenne: Prec=0.666, Recall=0.738, F1=0.700
# auto-svm: 
# MST Prec=0.636, Recall=0.706, F1=0.669


# dev+train
# - local Prec=0.502, Recall=0.487, F1=0.487
# - RFC beam(2):  Prec=0.639, Recall=0.630, F1=0.634
# - MST: Prec=0.687, Recall=0.678, F1=0.682
# - RFC h_moyenne: Prec=0.680, Recall=0.670, F1=0.675 (same as beam=10)


# LOO validation
# windowed
# local: Prec=0.632, Recall=0.634, F1=0.624
# RFC beam(2): Prec=0.614, Recall=0.677, F1=0.644




# default fold number
FOLDS_NB = 10
# leave-one-out : 66 files
#FOLDS_NB = 66



# the windowed instances miss a % of the reference
#RECALL_CORRECTION = 0.91
# the full data set
#RECALL_CORRECTION = 1.0



if __name__ == "__main__":
    import sys
    import optparse
    import pprint
    # usage: argv1 is attachment data file
    # if there is an argv2, it is the relation data file

    usage = "usage: %prog [options] attachement_data_file [relation_data_file]"
    parser = optparse.OptionParser(usage = usage)
    parser.add_option("-l", "--learners", default = "bayes",
                      help = "comma separated list of learners for attacht [and relations]; implemented: bayes, svm, maxent, perc, struc_perc; default (naive) bayes")
    parser.add_option("-d", "--decoders", default = "local",
                      help = "comma separated list of decoders for attacht [and relations]; implemented: local, last, mst, locallyGreedy, astar (cf also heuristics); default:local")
    parser.add_option("-e", "--heuristics", default = "average", type = "choice", choices = ["zero", "max", "best", "average"],
                      help = "heuristics used for astar decoding; default = average")
    parser.add_option("-n", "--nfold", default = FOLDS_NB, type = "int",
                      help = "nfold cross-validation number (default 10)")
    parser.add_option("-o", "--output", default = None,
                      help = "if this option is set to an existing path, predicted structures will be saved there; nothing saved otherwise")
    parser.add_option("-c", "--correction", default = 1.0, type = "float",
                      help = "if input is already a restriction on the full task, this options defines a correction to apply on the final recall score to have the real scores on the full corpus")
    parser.add_option("-t", "--threshold", default = None, type = "float",
                      help = "force the classifier to use this threshold value for attachment decisions, unless it is trained explicitely with a threshold")
    parser.add_option("-u", "--unlabelled", default = False, action = "store_true",
                      help = "force unlabelled evaluation, even if the prediction is made with relations")
    parser.add_option("-p", "--post-label", default = False, action = "store_true",
                      help = "decode only on attachment, and predict relations afterwards")
    parser.add_option("-r", "--rfc", default = "full",type = "choice", choices = ["full","simple","none"],
                      help = "with astar decoding, what kind of RFC is applied: simple of full; simple means everything is subordinating")
    parser.add_option("-a", "--accuracy", default = False, action = "store_true",
                      help = "provide accuracy scores for classifiers used")
    parser.add_option("-m", "--averaging", default = False, action = "store_true", help = "averaged perceptron")
    parser.add_option("-i", "--nit", default = 1, type = "int",help = "number of iterations for perceptron models")
    parser.add_option("-P", "--use_prob", default = True, action = "store_false", help = "convert perceptron scores into probabilities")
    parser.add_option("-s","--shuffle",default=False, action = "store_true",
                      help="if set, ensure a different cross-validation of files is done, otherwise, the same file splitting is done everytime")
    parser.add_option("-C", "--corpus", default = "annodis", type = "choice", choices = ["annodis","stac"],
                      help = "corpus type (annodis or stac), default: annodis")
    parser.add_option("-X", "--config", default = None,
                      help = "TEST OPTION: corpus specificities config file; if absent, defaults to hard-wired annodis config; when ok, should replace -C")
    # simple parser with separate train/test
    parser.add_option("-A", "--attachment-model", default = None, help = "provide saved model for prediction of attachment (only with -T option)")
    parser.add_option("-R", "--relation-model", default = None, help = "provide saved model for prediction of relations (only with -T option)")
    parser.add_option("-T", "--test-only", default = False, action = "store_true", help = "predicts on the given  data (requires a model for -A option or two with -A and -R option), save to output directory, forces -o option is not set with output/ as default path; does not make any evaluation, even if the class labels are present")
    parser.add_option("-S", "--save-models", default = False, action = "store_true", help = "train on the whole instance set provided, and save attachment [and relation] models to attach.model and relation.model")

    # todo for options 
    # RFC type 
    # beam size
    # nbest eval (when implemented)
    # 

    (options, args) = parser.parse_args()

    output_folder = options.output
    # todo: test existence; create if needed

    if options.config is not None: 
        config = ConfigParser()
        # cancels case-insensitive reading of variables. 
        config.optionxform = lambda option: option
        config.readfp(open(options.config))
        metacfg = dict(config.items("Meta features"))
    elif options.corpus.lower() == "stac":
        metacfg = stac_cfg
    else:# annodis config as default, should not cause regression on coling experiment
        metacfg =  def_cfg

    # index names for EDU pairs 
    # if options.corpus.lower() == "stac": 
    #     FirstNode = "id_DU1"
    #     SecondNode = "id_DU2"
    #     TargetSpanStart = "start_DU2"
    #     TargetSpanEnd = "end_DU2"
    #     SourceSpanStart = "start_DU1"
    #     SourceSpanEnd = "end_DU1"
    #     FILE = "document"
 

    data_attach = Orange.data.Table(args[0])
    # print "DATA ATTACH:", data_attach
    
    if len(args) > 1:
        data_relations = Orange.data.Table(args[1])
        with_relations = True
    else:
        data_relations = None
        with_relations = False
        labels = None
        if options!="none": options.rfc = "simple"



    # decoders 
    _heuristics = {"average":h_average, "best":h_best, "max":h_max, "zero":h0}
    heuristic = _heuristics.get(options.heuristics, h_average)
    _decoders = {"last":last_baseline,"local":local_baseline, "locallyGreedy":locallyGreedy, "mst":MST_decoder, "astar":lambda x, **kargs: astar_decoder(x, heuristics = heuristic, RFC = options.rfc, **kargs)}
    all_decoders = [_decoders.get(x, local_baseline) for x in options.decoders.split(",")]

    # orange classifiers
    bayes = Orange.classification.bayes.NaiveLearner(adjust_threshold = True)
    bayes.name = "naive bayes"
    #svm = Orange.classification.svm.SVMLearnerEasy(probability = True)
    svm = Orange.classification.svm.SVMLearner(probability = True)
    svm.name = "svm"
    maxent = MaxentLearner() #Orange.classification.logreg.LogRegLearner()
    maxent.name = "maxent"
    majority = Orange.classification.majority.MajorityLearner()
    majority.name = "majority"

    # home made perceptron 
    perc = Perceptron( nber_it=options.nit, avg=options.averaging , cfg = metacfg) 
    # home made structured perceptron 
    struc_perc = StructuredPerceptron( all_decoders[0], nber_it=options.nit, avg=options.averaging  , cfg = metacfg) 
    
    _learners = {"bayes":bayes, "svm":svm, "maxent":maxent,"majority":majority, "perc":perc, "struc_perc":struc_perc}
    all_learners = [_learners.get(x, bayes) for x in options.learners.split(",")]
    
    RECALL_CORRECTION = options.correction


    # id for EDU  (isn't it just the index?)
    arg1 = data_attach.domain.index(metacfg["FirstNode"])
    arg2 = data_attach.domain.index(metacfg["SecondNode"])

    # indices for spans and file
    TargetSpanStartIndex = data_attach.domain.index(metacfg["TargetSpanStart"])
    TargetSpanEndIndex = data_attach.domain.index(metacfg["TargetSpanEnd"])
    SourceSpanStartIndex = data_attach.domain.index(metacfg["SourceSpanStart"])
    SourceSpanEndIndex = data_attach.domain.index(metacfg["SourceSpanEnd"])
    FILEIndex = data_attach.domain.index(metacfg["FILE"])


    # prepare n-fold-by-file
    import random
    if options.shuffle:
        random.seed()
    else:
        random.seed("just an illusion")

    if options.save_models or options.test_only:# training only or testing only => no folds
        options.nfold = 1

    fold_struct = make_n_fold(data_attach, folds = options.nfold,meta_index=metacfg["FILE"])
    
    selection = makeFoldByFileIndex(data_attach, fold_struct,meta_index=metacfg["FILE"])
    # only one learner+decoder for now
    learner = all_learners[0]
    decoder = all_decoders[0]

    use_threshold = options.threshold is not None
    # eval procedures
    if options.test_only: 
        structure_eval = lambda x,y, labels=None: 0
    else:
        structure_eval = lambda x,y, labels=None: discourse_eval(x,y, cfg = metacfg, labels=labels)
    save_results = options.output is not None

    # TODO: refactor from here, using above as parameters
    evals = []
    # --- fold level -- to be refactored
    for test_fold in range(options.nfold):
        print >> sys.stderr, ">>> doing fold ", test_fold + 1
        if not(options.test_only):
            print >> sys.stderr, ">>> training ... "
            if options.save_models:# training only
                train_data_attach = data_attach.select_ref(selection, test_fold)
            else:
                train_data_attach = data_attach.select_ref(selection, test_fold, negate = 1)
            # train model
            if options.learners == "struc_perc":
                model = learner(train_data_attach, use_prob=options.use_prob)
            else:
                model = learner(train_data_attach)
            if options.save_models:# training only
                attm = open("attach.model","wb")
                cPickle.dump(model,attm)
                attm.close()
        else:# test-only
            if options.attachment_model is None:
                print >> sys.stderr, "ERROR, attachment model not provided with -A"
                sys.exit(0)
            attm = open(options.attachment_model,"rb")
            model = cPickle.load(attm)
           
        if use_threshold or str(decoder.__name__) == "local_baseline":
            try:
                threshold = model.threshold
            except:
                print >> sys.stderr, "treshold forced at : ",  options.threshold
                threshold = options.threshold if use_threshold else 0.5
        else:
            threshold = None
 

        # test 
        if not(options.save_models):# else would be training only
            test_data_attach = data_attach.select_ref(selection, test_fold)
        # 
        if with_relations  and not(options.test_only):
            if options.save_models:
                train_data_relations = data_relations.select_ref(selection, test_fold)
            else:
                train_data_relations = data_relations.select_ref(selection, test_fold, negate = 1)
            train_data_relations = train_data_relations.filter_ref({"CLASS":["UNRELATED"]}, negate = 1)
            # train model
            model_relations = learner(train_data_relations)
            if options.save_models:# training only
                relm = open("relations.model","wb")
                cPickle.dump(model_relations,relm)
                relm.close()
        elif with_relations and not(options.save_models):
            test_data_relations = data_relations.select_ref(selection, test_fold)
            if options.test_only:
                relm=open(options.relation_model,"rb")
                model_relations = cPickle.load(relm)
        else:# no relations
            model_relations = None
        if options.save_models:# training done, leaving
            print >> sys.stderr, "done with training, exiting"
            sys.exit(0)
        # -- file level --
        fold_evals = []
        for onedoc in fold_struct:
            if fold_struct[onedoc] == test_fold:
                print >> sys.stderr, "decoding on file : ", onedoc
                scores = process_document(onedoc, model, decoder, data_attach,
                                          with_relations = with_relations,
                                          data_relations = data_relations,
                                          model_relations = model_relations,
                                          save_results = save_results,
                                          output_folder = output_folder,
                                          threshold = threshold,
                                          unlabelled = options.unlabelled,
                                          post_labelling = options.post_label,
                                          use_prob = options.use_prob , cfg = metacfg )
                if not(options.test_only): 
                    evals.append(scores)
                    fold_evals.append(scores)
        options.relations = ["attach","relations"][with_relations]
        options.context = "window5" if "window" in args[0] else "full"
        options.relnb = args[1].split(".")[-2][-6:] if with_relations else "-"
        if not(options.test_only): 
            fold_report = Report(fold_evals, params = options, correction = RECALL_CORRECTION)
            print "Fold eval:", fold_report.summary()
        # --end of file level
       # --- end of fold level
    # end of test for a set of parameter
    # report: summing : TODO: must register many runs with change of parameters
    options.relations = ["attach","relations"][with_relations]
    options.context = "window5" if "window" in args[0] else "full"
    options.relnb = args[1].split(".")[-2][-6:] if with_relations else "-"
    if not(options.test_only): 
        report = Report(evals, params = options, correction = RECALL_CORRECTION)
        print ">>> FINAL EVAL:", report.summary()
        report.save("results/"+"{relations}_{context}_{relnb}_{decoders}_{learners}_{heuristics}_{unlabelled}_{post_label}_{rfc}".format(**options.__dict__))



# full train test -- 4 relations sdrt

#  attach+relation
# A*+average: Prec=0.382, Recall=0.377, F1=0.380
# with full RFC Prec=0.394, Recall=0.388, F1=0.391

# MST: Prec=0.264, Recall=0.260, F1=0.262
# local: to be done, threshold fucked-up by prob. combination. 
# with arbitray 0.5: Prec=0.376, Recall=0.316, F1=0.343

# same, unlabelled eval (!! not the same as attchmt prediction only)
# a*: Prec=0.571, Recall=0.564, F1=0.568
# mst: Prec=0.575, Recall=0.568, F1=0.571
# local  Prec=0.520, Recall=0.437, F1=0.475

# attachmt pred only
# a*: Prec=0.618, Recall=0.610, F1=0.614
# mst: Prec=0.624, Recall=0.616, F1=0.620
# local: Prec=0.670, Recall=0.464, F1=0.548

# window 5
# -- local attacht 
#Prec=0.770, Recall=0.557, F1=0.647
#with recall correction estimate (0.91%), F1=0.611496928675
# -- mst Prec=0.667, Recall=0.742, F1=0.702
#with recall correction estimate, F1=0.670959965492
# -- a* Prec=0.663, Recall=0.737, F1=0.698
# with recall correction estimate, F1=0.666557822471

# with relations

