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

from ParseSearch import astar_decoder, h0, h_best, h_max, h_average
from attachment.baseline import local_baseline, last_baseline
from attachment.mst import MST_list_edges as MST_decoder
from attachment.greedy import locallyGreedy, getSortedEDUs

from megam_wrapper import MaxentLearner
from online_learner import Perceptron, StructuredPerceptron

from edu         import EDU, mk_edu_pairs
from features    import Features

from fileNfold import make_n_fold, makeFoldByFileIndex
from report      import Report

# from MST import MSTdecoder

annodis_features=Features() # default settings
stac_features=Features(source="id_DU1",
                       target="id_DU2",
                       source_span_start = "start_DU1",
                       source_span_end   = "end_DU1",
                       target_span_start = "start_DU2",
                       target_span_end   = "end_DU2",
                       grouping          = "dialogue",
                       label             = "CLASS")


def discourse_eval(features, predicted, data, labels = None, debug = False):
    """basic eval: counting correct predicted edges (labelled or not)
    data contains the reference attachments
    labels the corresponding relations
    """
    #print "REF:", data
    #print "PRED:", predicted
    score = 0
    dict_predicted = dict([((a1, a2), rel) for (a1, a2, rel) in predicted])
    for one in data:
        arg1 = one[features.source].value
        arg2 = one[features.target].value
        if debug:
            print >> sys.stderr, arg1, arg2, dict_predicted.get((arg1, arg2))
        if dict_predicted.has_key((arg1, arg2)):
            if labels is None:
                score += 1
                if debug: print >> sys.stderr, "correct"
            else:
                relation_ref = labels.filter_ref({features.source:[arg1], features.target:[arg2]})
                if len(relation_ref) == 0:
                    print >> sys.stderr, "attached pair without corresponding relation", one[features.grouping], arg1, arg2
                else:
                    relation_ref = relation_ref[0][features.label].value
                    score += (dict_predicted[(arg1, arg2)] == relation_ref)
    #print "SCORE:", score
    return score

def combine_probs(features, attach_instances, relation_instances, attachmt_model, relations_model):
    """retrieve probability of the best relation on an edu pair, given the probability of an attachment
    """
    # !! instances set must correspond to same edu pair in the same order !!
    distrib = []

    edu_pair           = mk_edu_pairs(features, attach_instances.domain)
    attach_instances   = sorted(attach_instances,   key=lambda x:x.get_metas())
    relation_instances = sorted(relation_instances, key=lambda x:x.get_metas())

    for i, (attach, relation) in enumerate(zip(attach_instances, relation_instances)):
        p_attach    = attachmt_model(attach, Orange.classification.Classifier.GetProbabilities)[1]
        p_relations = relations_model(relation, Orange.classification.Classifier.GetBoth)
        if not instance_check(features, attach, relation):
            print >> sys.stderr, "mismatch of attacht/relation instance, instance number", i,
            meta_info(attach,   features),
            meta_info(relation, features)
        # this should be investigated
        try:
            best_rel = p_relations[0].value
        except:
            best_rel = p_relations[0]

        rel_prob   = max(p_relations[1])
        edu1, edu2 = edu_pair(attach)
        distrib.append((edu1, edu2, p_attach * rel_prob, best_rel))
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



def add_labels(features, predicted, rel_instances, relations_model):
    """ predict labels for a given set of edges (=post-labelling an unlabelled decoding)
    """
    rels = index_by_metas(rel_instances,metas=[features.source,features.target])
    result = []
    for (a1,a2,_r) in predicted:
        instance_rel = rels[(a1,a2)]
        rel = relations_model(instance_rel,Orange.classification.Classifier.GetValue)
        result.append((a1,a2,rel))
    return result

def instance_check(features, one, two):
    return  one[features.source]   == two[features.source] and\
            one[features.target]   == two[features.target] and\
            one[features.grouping] == two[features.grouping]

def meta_info(features, instance):
    return "%s: %s-%s" % (instance[features.grouping], instance[features.source], instance[features.target])


def exportGraph(predicted, doc, folder):
    fname = os.path.join(folder, doc + ".rel")
    if not os.path.exists(folder):
        os.makedirs(folder)
    f = open(fname, 'w')
    for (a1, a2, rel) in predicted:
        f.write(rel + " ( " + a1 + " / " + a2 + " )\n")
    f.close()



def process_document(features, onedoc, model, decoder, data_attach,
                     structure_eval,
                     with_relations = False, data_relations = [], model_relations = None,
                     save_results = False, output_folder = None,
                     threshold = None,
                     unlabelled = False,
                     post_labelling=False,
                     use_prob=True):
    """decode one document (onedoc), selecting instances for attachment from data_attach, (idem relations if present),
    using trained model,model

    TODO: check that call to learner can be uniform with 2 parameters (as logistic), as the documentation is inconsistent on this
    """


    FILE  = features.grouping
    CLASS = features.label

    doc_instances = data_attach.filter_ref({FILE : onedoc})

    if with_relations:
        rel_instances = data_relations.filter_ref({FILE : onedoc})
    if with_relations and not(post_labelling):
        prob_distrib = combine_probs(features, doc_instances, rel_instances, model, model_relations)
    else:
        # home-made online models
        if model.name in ["Perceptron", "StructuredPerceptron"]:
            prob_distrib = model.get_scores( doc_instances, use_prob=use_prob )
        # orange-based models
        else:
            edu_pair     = mk_edu_pairs(features, doc_instances.domain)
            prob_distrib = []
            for one in doc_instances:
                edu1, edu2 = edu_pair(one)
                probs      = model(one, Orange.classification.Classifier.GetProbabilities)[1]
                prob_distrib.append((edu1, edu2, probs, "unlabelled"))
    # print prob_distrib

    # get prediction (input is just prob_distrib)
    if threshold:
        predicted = decoder(prob_distrib, threshold = threshold, use_prob=use_prob)
        # predicted = decoder(prob_distrib, threshold = threshold)
    else:
        predicted = decoder(prob_distrib, use_prob=use_prob)
        # predicted = decoder(prob_distrib)

    if post_labelling:
        predicted = add_labels(features, predicted, rel_instances, model_relations, use_prob=use_prob)
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

def main():
    import sys
    import argparse
    import pprint
    # usage: argv1 is attachment data file
    # if there is an argv2, it is the relation data file

    usage = "%(prog)s [options] attachment_data_file [relation_data_file]"
    parser = argparse.ArgumentParser(usage = usage)
    parser.add_argument("data_attach",    metavar="FILE",
                        help="attachment data")
    parser.add_argument("data_relations", metavar="FILE", nargs="?",
                        help="relations data") # optional
    parser.add_argument("--learners", "-l", default = "bayes",
                        help = "comma separated list of learners for attacht [and relations]; implemented: bayes, svm, maxent, perc, struc_perc; default (naive) bayes")
    parser.add_argument("--decoders", "-d", default = "local",
                        help = "comma separated list of decoders for attacht [and relations]; implemented: local, last, mst, locallyGreedy, astar (cf also heuristics); default:local")
    parser.add_argument("--heuristics", "-e", default = "average", choices = ["zero", "max", "best", "average"],
                        help = "heuristics used for astar decoding; default = average")
    parser.add_argument("--nfold", "-n", default = 10, type = int,
                        help = "nfold cross-validation number (default 10)")
    parser.add_argument("--output", "-o", default = None,
                        help = "if this option is set to an existing path, predicted structures will be saved there; nothing saved otherwise")
    parser.add_argument("--correction", "-c", default = 1.0, type = float,
                        help = "if input is already a restriction on the full task, this options defines a correction to apply on the final recall score to have the real scores on the full corpus")
    parser.add_argument("--threshold", "-t", default = None, type = float,
                        help = "force the classifier to use this threshold value for attachment decisions, unless it is trained explicitely with a threshold")
    parser.add_argument("--unlabelled", "-u", default = False, action = "store_true",
                        help = "force unlabelled evaluation, even if the prediction is made with relations")
    parser.add_argument("--post-label", "-p", default = False, action = "store_true",
                        help = "decode only on attachment, and predict relations afterwards")
    parser.add_argument("--rfc", "-r", default = "full", choices = ["full","simple","none"],
                        help = "with astar decoding, what kind of RFC is applied: simple of full; simple means everything is subordinating")
    parser.add_argument("--accuracy", "-a", default = False, action = "store_true",
                        help = "provide accuracy scores for classifiers used")
    parser.add_argument("--averaging", "-m", default = False, action = "store_true",
                        help = "averaged perceptron")
    parser.add_argument("--nit", "-i", default = 1, type = int,
                        help = "number of iterations for perceptron models")
    parser.add_argument("--use_prob", "-P", default = True, action = "store_false",
                        help = "convert perceptron scores into probabilities")
    parser.add_argument("-s","--shuffle",default=False, action = "store_true",
                        help="if set, ensure a different cross-validation of files is done, otherwise, the same file splitting is done everytime")
    parser.add_argument("--corpus", "-C", default = "annodis", choices = ["annodis","stac"],
                        help = "corpus type (annodis or stac), default: annodis")
    parser.add_argument("--config", "-X", default = None,
                        help = "TEST OPTION: corpus specificities config file; if absent, defaults to hard-wired annodis config; when ok, should replace -C")
    # simple parser with separate train/test
    parser.add_argument("--attachment-model", "-A", default = None,
                        help = "provide saved model for prediction of attachment (only with -T option)")
    parser.add_argument("--relation-model", "-R", default = None,
                        help = "provide saved model for prediction of relations (only with -T option)")
    parser.add_argument("--test-only", "-T", default = False, action = "store_true",
                        help = "predicts on the given  data (requires a model for -A option or two with -A and -R option), save to output directory, forces -o option is not set with output/ as default path; does not make any evaluation, even if the class labels are present")
    parser.add_argument("--save-models", "-S", default = False, action = "store_true",
                        help = "train on the whole instance set provided, and save attachment [and relation] models to attach.model and relation.model")

    # todo for options
    # RFC type
    # beam size
    # nbest eval (when implemented)
    #

    args = parser.parse_args()

    output_folder = args.output
    # todo: test existence; create if needed

    if args.config is not None:
        config = ConfigParser()
        # cancels case-insensitive reading of variables.
        config.optionxform = lambda option: option
        config.readfp(open(args.config))
        metacfg = dict(config.items("Meta features"))
        features = Dataset(source=metacfg["FirstNode"],
                          target=metacfg["SecondNode"],
                          source_span_start=metacfg["SourceSpanStart"],
                          source_span_end=metacfg["SourceSpanEnd"],
                          target_span_start=metacfg["TargetSpanStart"],
                          target_span_end=metacfg["TargetSpanEnd"],
                          grouping=metacfg["FILE"],
                          label=metacfg["CLASS"])
    elif args.corpus.lower() == "stac":
        features = stac_features
    else:# annodis config as default, should not cause regression on coling experiment
        features = annodis_features

    data_attach = Orange.data.Table(args.data_attach)
    # print "DATA ATTACH:", data_attach

    if args.data_relations:
        data_relations = Orange.data.Table(args.data_relations)
        with_relations = True
    else:
        data_relations = None
        with_relations = False
        labels = None
        args.rfc = "simple"



    # decoders
    _heuristics = {"average":h_average, "best":h_best, "max":h_max, "zero":h0}
    heuristic = _heuristics.get(args.heuristics, h_average)
    _decoders = {"last":last_baseline,"local":local_baseline, "locallyGreedy":locallyGreedy, "mst":MST_decoder, "astar":lambda x, **kargs: astar_decoder(x, heuristics = heuristic, RFC = args.rfc, **kargs)}
    all_decoders = [_decoders.get(x, local_baseline) for x in args.decoders.split(",")]

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
    perc = Perceptron( features=features, nber_it=args.nit, avg=args.averaging )
    # home made structured perceptron
    struc_perc = StructuredPerceptron( features, all_decoders[0], nber_it=args.nit, avg=args.averaging )

    _learners = {"bayes":bayes, "svm":svm, "maxent":maxent,"majority":majority, "perc":perc, "struc_perc":struc_perc}
    all_learners = [_learners.get(x, bayes) for x in args.learners.split(",")]

    RECALL_CORRECTION = args.correction

    # prepare n-fold-by-file
    import random
    if args.shuffle:
        random.seed()
    else:
        random.seed("just an illusion")

    if args.save_models or args.test_only:# training only or testing only => no folds
        args.nfold = 1

    fold_struct = make_n_fold(data_attach, folds = args.nfold,meta_index=features.grouping)

    selection = makeFoldByFileIndex(data_attach, fold_struct,meta_index=features.grouping)

    # only one learner+decoder for now
    learner = all_learners[0]
    decoder = all_decoders[0]

    use_threshold = args.threshold is not None
    # eval procedures
    if args.test_only:
        structure_eval = lambda x,y, labels=None: 0
    else:
        structure_eval = lambda x,y, labels=None: discourse_eval(features, x,y, labels=labels)
    save_results = args.output is not None

    # TODO: refactor from here, using above as parameters
    evals = []
    # --- fold level -- to be refactored
    for test_fold in range(args.nfold):
        print >> sys.stderr, ">>> doing fold ", test_fold + 1
        if not(args.test_only):
            print >> sys.stderr, ">>> training ... "
            if args.save_models:# training only
                train_data_attach = data_attach.select_ref(selection, test_fold)
            else:
                train_data_attach = data_attach.select_ref(selection, test_fold, negate = 1)
            # train model
            if args.learners == "struc_perc":
                model = learner(train_data_attach, use_prob=args.use_prob)
            else:
                model = learner(train_data_attach)
            if args.save_models:# training only
                attm = open("attach.model","wb")
                cPickle.dump(model,attm)
                attm.close()
        else:# test-only
            if args.attachment_model is None:
                print >> sys.stderr, "ERROR, attachment model not provided with -A"
                sys.exit(0)
            attm = open(args.attachment_model,"rb")
            model = cPickle.load(attm)

        if use_threshold or str(decoder.__name__) == "local_baseline":
            try:
                threshold = model.threshold
            except:
                print >> sys.stderr, "treshold forced at : ",  args.threshold
                threshold = args.threshold if use_threshold else 0.5
        else:
            threshold = None


        # test
        if not(args.save_models):# else would be training only
            test_data_attach = data_attach.select_ref(selection, test_fold)
        #
        if with_relations  and not(args.test_only):
            if args.save_models:
                train_data_relations = data_relations.select_ref(selection, test_fold)
            else:
                train_data_relations = data_relations.select_ref(selection, test_fold, negate = 1)
            train_data_relations = train_data_relations.filter_ref({"CLASS":["UNRELATED"]}, negate = 1)
            # train model
            model_relations = learner(train_data_relations)
            if args.save_models:# training only
                relm = open("relations.model","wb")
                cPickle.dump(model_relations,relm)
                relm.close()
        elif with_relations and not(args.save_models):
            test_data_relations = data_relations.select_ref(selection, test_fold)
            if args.test_only:
                relm=open(args.relation_model,"rb")
                model_relations = cPickle.load(relm)
        else:# no relations
            model_relations = None
        if args.save_models:# training done, leaving
            print >> sys.stderr, "done with training, exiting"
            sys.exit(0)
        # -- file level --
        fold_evals = []
        for onedoc in fold_struct:
            if fold_struct[onedoc] == test_fold:
                print >> sys.stderr, "decoding on file : ", onedoc
                scores = process_document(features,
                                          onedoc, model, decoder, data_attach,
                                          structure_eval,
                                          with_relations = with_relations,
                                          data_relations = data_relations,
                                          model_relations = model_relations,
                                          save_results = save_results,
                                          output_folder = output_folder,
                                          threshold = threshold,
                                          unlabelled = args.unlabelled,
                                          post_labelling = args.post_label,
                                          use_prob = args.use_prob)
                if not(args.test_only):
                    evals.append(scores)
                    fold_evals.append(scores)
        args.relations = ["attach","relations"][with_relations]
        args.context = "window5" if "window" in args.data_attach else "full"
        args.relnb = args.data_relations.split(".")[-2][-6:] if with_relations else "-"
        if not(args.test_only):
            fold_report = Report(fold_evals, params = args, correction = RECALL_CORRECTION)
            print "Fold eval:", fold_report.summary()
        # --end of file level
       # --- end of fold level
    # end of test for a set of parameter
    # report: summing : TODO: must register many runs with change of parameters
    args.relations = ["attach","relations"][with_relations]
    args.context = "window5" if "window" in args.data_attach else "full"
    args.relnb = args.data_relations.split(".")[-2][-6:] if with_relations else "-"
    if not(args.test_only):
        report = Report(evals, params = args, correction = RECALL_CORRECTION)
        print ">>> FINAL EVAL:", report.summary()
        report.save("results/"+"{relations}_{context}_{relnb}_{decoders}_{learners}_{heuristics}_{unlabelled}_{post_label}_{rfc}".format(**args.__dict__))

if __name__ == "__main__":
    main()
