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

import argparse
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

from attelo.decoding.astar    import astar_decoder, h0, h_best, h_max, h_average
from attelo.decoding.baseline import local_baseline, last_baseline
from attelo.decoding.mst      import MST_list_edges as MST_decoder
from attelo.decoding.greedy   import locallyGreedy, getSortedEDUs

from attelo.learning.megam      import MaxentLearner
from attelo.learning.perceptron import Perceptron, StructuredPerceptron

from attelo.edu         import EDU, mk_edu_pairs
from attelo.features    import Features

from attelo.fileNfold   import make_n_fold, makeFoldByFileIndex
from attelo.report      import Report

# from MST import MSTdecoder

ANNODIS_FEATURES=Features() # default settings

# ---------------------------------------------------------------------
# utilities
# ---------------------------------------------------------------------

def related_attachments(features, table):
    """Return just the entries in the attachments table that
    represent related EDU pairs
    """
    return table.filter_ref({features.label:"True"})

def related_relations(features, table):
    """Return just the entries in the relations table that represent
    related EDU pair
    """
    return table.filter_ref({features.label:["UNRELATED"]}, negate = 1)

def entries_in_grouping(features, grouping, table):
    """Return the entries in the table that belong in the given
    group
    """
    return table.filter_ref({features.grouping:grouping})

# ---------------------------------------------------------------------
#
# ---------------------------------------------------------------------

def load_model(filename):
    with open(filename, "rb") as f:
        return cPickle.load(f)

def save_model(filename, model):
    with open(filename, "wb") as f:
        cPickle.dump(model, f)

# ---------------------------------------------------------------------
#
# ---------------------------------------------------------------------

def discourse_eval(features, predicted, reference, labels = None, debug = False):
    """basic eval: counting correct predicted edges (labelled or not)
    data contains the reference attachments
    labels the corresponding relations
    """
    #print "REF:", reference
    #print "PRED:", predicted
    score = 0
    dict_predicted = dict([((a1, a2), rel) for (a1, a2, rel) in predicted])
    for one in reference:
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

    total_ref  = len(reference)
    total_pred = len(predicted)
    return score, total_pred, total_ref

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
            print >> sys.stderr, "mismatch of attachment/relation instance, instance number", i,\
                    meta_info(attach,   features),\
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
    """
    Return True if the two annotations should be considered as refering to the same
    EDU pair. This can be used as a sanity check when zipping two datasets that are
    expected to be on the same EDU pairs.
    """
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

# ---------------------------------------------------------------------
# learning
# ---------------------------------------------------------------------

class LearnerConfig(object):
    def __init__(self, use_prob):
        self.use_prob = use_prob

def learn_attachments(config, learner, data):
    # TODO: is it intentional for use_prob to only be used
    # - for StructuredPerceptron and not Perceptron?
    # - for attachment and not labelling?
    #
    # If it's not intentional, I'd like to make use_prob a
    # parameter of the perceptron class, and then kill these
    # two functions
    if learner.name == "StructuredPerceptron":
        return learner(data, use_prob=config.use_prob)
    else:
        return learner(data)

def learn_relations(config, learner, data):
    return learner(data)

# ---------------------------------------------------------------------
# processing a single document
# ---------------------------------------------------------------------

class DecoderConfig(object):
    def __init__(self, features, model_attach, model_relations, decoder,
                 threshold=None,
                 post_labelling=False,
                 use_prob=True):
        self.features         = features
        self.decoder          = decoder
        self.model_attach     = model_attach
        self.model_relations  = model_relations
        self.threshold        = threshold
        self.post_labelling   = post_labelling
        self.use_prob         = use_prob

def decode_document(config, attach_instances, rel_instances=None):
    """
    decode one document (onedoc), selecting instances for attachment from data_attach, (idem relations if present),
    using trained model,model

    Return the predictions made

    TODO: check that call to learner can be uniform with 2 parameters (as logistic), as the documentation is inconsistent on this
    """
    features        = config.features
    decoder         = config.decoder
    model           = config.model_attach
    model_relations = config.model_relations
    threshold       = config.threshold
    use_prob        = config.use_prob

    if rel_instances and not config.post_labelling:
        prob_distrib = combine_probs(features, attach_instances, rel_instances, model, model_relations)
    elif model.name in ["Perceptron", "StructuredPerceptron"]:
        # home-made online models
        prob_distrib = model.get_scores( attach_instances, use_prob=use_prob )
    else:
        # orange-based models
        edu_pair     = mk_edu_pairs(features, attach_instances.domain)
        prob_distrib = []
        for one in attach_instances:
            edu1, edu2 = edu_pair(one)
            probs      = model(one, Orange.classification.Classifier.GetProbabilities)[1]
            prob_distrib.append((edu1, edu2, probs, "unlabelled"))
    # print prob_distrib

    # get prediction (input is just prob_distrib)
    if threshold:
        predicted = decoder(prob_distrib, threshold=threshold, use_prob=use_prob)
        # predicted = decoder(prob_distrib, threshold = threshold)
    else:
        predicted = decoder(prob_distrib, use_prob=use_prob)
        # predicted = decoder(prob_distrib)

    if config.post_labelling:
        predicted = add_labels(features, predicted, rel_instances, model_relations)
        # predicted = add_labels(predicted, rel_instances, model_relations)

    return predicted

# ---------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------

def prepare_folds(features, num_folds, table, shuffle=True):
    """Return an N-fold validation setup respecting a property where
    examples in the same grouping stay in the same fold.
    """
    import random
    if shuffle:
        random.seed()
    else:
        random.seed("just an illusion")

    fold_struct = make_n_fold(table, folds=num_folds, meta_index=features.grouping)
    selection = makeFoldByFileIndex(table, fold_struct, meta_index=features.grouping)
    return fold_struct, selection

# ---------------------------------------------------------------------
# arguments and main
# ---------------------------------------------------------------------

def args_to_features(args):
    """
    Given the (parsed) command line arguments, return the set of
    core feature labels for our incoming dataset.

    If no configuration file is provided, we default to the
    Annodis experiment settings
    """
    if args.config:
        config = ConfigParser()
        # cancels case-insensitive reading of variables.
        config.optionxform = lambda option: option
        with open(args.config) as config_file:
            config.readfp(config_file)
            metacfg = dict(config.items("Meta features"))
            return Features(source=metacfg["FirstNode"],
                            target=metacfg["SecondNode"],
                            source_span_start=metacfg["SourceSpanStart"],
                            source_span_end=metacfg["SourceSpanEnd"],
                            target_span_start=metacfg["TargetSpanStart"],
                            target_span_end=metacfg["TargetSpanEnd"],
                            grouping=metacfg["Grouping"],
                            label=metacfg["Label"])
    else:# annodis config as default, should not cause regression on coling experiment
        return ANNODIS_FEATURES

def args_to_decoders(args):
    """
    Given the (parsed) command line arguments, return a sequence of
    decoders in the order they were requested on the command line
    """
    _heuristics = {"average":h_average,
                   "best":h_best,
                   "max":h_max,
                   "zero":h0}

    if args.heuristics not in _heuristics:
        raise argparse.ArgumentTypeError("Unknown heuristics: %s" % args.heuristics)
    heuristic = _heuristics.get(args.heuristics, h_average)

    if not args.data_relations:
        args.rfc = "simple"

    _decoders = {"last":last_baseline,
                 "local":local_baseline,
                 "locallyGreedy":locallyGreedy,
                 "mst":MST_decoder,
                 "astar":lambda x, **kargs: astar_decoder(x, heuristics = heuristic, RFC = args.rfc, **kargs)}

    requests = args.decoders.split(",")
    unknown  = ",".join(r for r in requests if r not in _decoders)
    if unknown:
        raise argparse.ArgumentTypeError("Unknown decoders: %s" % unknown)
    else:
        return [_decoders[x] for x in requests]

def args_to_learners(decoder, features, args):
    """
    Given the (parsed) command line arguments, return a sequence of
    learners in the order they were requested on the command line
    """

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
    struc_perc = StructuredPerceptron(features, decoder,
                                      nber_it=args.nit,
                                      avg=args.averaging )

    _learners = {"bayes":bayes,
                 "svm":svm,
                 "maxent":maxent,
                 "majority":majority,
                 "perc":perc,
                 "struc_perc":struc_perc}

    requests = args.learners.split(",")
    unknown  = ",".join(r for r in requests if r not in _learners)
    if unknown:
        raise argparse.ArgumentTypeError("Unknown learners: %s" % unknown)
    else:
        return [_learners[x] for x in requests]

def args_to_threshold(model, decoder, requested=None, default=0.5):
    """Given a model and decoder, return a threshold if

    * we request a specific threshold
    * or the decoder absolutely requires one

    In these cases, we try to return one of the following thresholds
    in order:

    1. that supplied by the model (if there is one)
    2. the requested threshold (if supplied)
    3. a default value
    """
    if requested or str(decoder.__name__) == "local_baseline":
        try:
            threshold = model.threshold
        except:
            threshold = requested if requested else default
            print >> sys.stderr, "threshold forced at : ", threshold
    else:
        threshold = None
    return threshold

def main():
    # usage: argv1 is attachment data file
    # if there is an argv2, it is the relation data file

    usage="%(prog)s [options] attachment_data_file [relation_data_file]"
    parser=argparse.ArgumentParser(usage=usage)
    parser.add_argument("data_attach",    metavar="FILE",
                        help="attachment data")
    parser.add_argument("data_relations", metavar="FILE", nargs="?",
                        help="relations data") # optional
    parser.add_argument("--config", "-C", metavar="FILE",
                        default=None,
                        help="corpus specificities config file; if absent, defaults to hard-wired annodis config")

    parser.add_argument("--learners", "-l",
                        default="bayes",
                        help="comma separated list of learners for attacht [and relations]; implemented: bayes, svm, maxent, perc, struc_perc; default (naive) bayes")
    parser.add_argument("--decoders", "-d", default="local",
                        help="comma separated list of decoders for attacht [and relations]; implemented: local, last, mst, locallyGreedy, astar (cf also heuristics); default:local")

    # classifier prefs
    classifier_group = parser.add_argument_group('classifier arguments')
    classifier_group.add_argument("--threshold", "-t",
                                  default=None, type=float,
                                  help="force the classifier to use this threshold value for attachment decisions, unless it is trained explicitely with a threshold")
    ## classifier prefs (perceptron)
    classifier_group.add_argument("--averaging", "-m",
                                  default=False, action="store_true",
                                  help="averaged perceptron")
    classifier_group.add_argument("--nit", "-i",
                                  default=1, type=int,
                                  help="number of iterations for perceptron models")
    classifier_group.add_argument("--use_prob", "-P",
                                  default=True, action="store_false",
                                  help="convert perceptron scores into probabilities")

    # decoder prefs
    decoder_group = parser.add_argument_group('decoder arguments')
    decoder_group.add_argument("--heuristics", "-e",
                               default="average", choices=["zero", "max", "best", "average"],
                               help="heuristics used for astar decoding; default=average")
    decoder_group.add_argument("--rfc", "-r",
                               default="full", choices=["full","simple","none"],
                               help="with astar decoding, what kind of RFC is applied: simple of full; simple means everything is subordinating")

    # harness prefs
    parser.add_argument("--output", "-o", default=None,
                        help="if this option is set to an existing path, predicted structures will be saved there; nothing saved otherwise")
    parser.add_argument("--post-label", "-p", default=False, action="store_true",
                        help="decode only on attachment, and predict relations afterwards")
    parser.add_argument("--attachment-model", "-A", default=None,
                        help="provide saved model for prediction of attachment (only with -T option)")
    parser.add_argument("--relation-model", "-R", default=None,
                        help="provide saved model for prediction of relations (only with -T option)")

    # commands
    parser.add_argument("--test-only", "-T", default=False, action="store_true",
                        help="predicts on the given  data (requires a model for -A option or two with -A and -R option), save to output directory, forces -o option is not set with output/ as default path; does not make any evaluation, even if the class labels are present")
    parser.add_argument("--save-models", "-S", default=False, action="store_true",
                        help="train on the whole instance set provided, and save attachment [and relation] models to attach.model and relation.model")

    # scoring profs
    evaluation_group = parser.add_argument_group('evaluation/scoring arguments')
    evaluation_group.add_argument("--nfold", "-n",
                                  default=10, type=int,
                                  help="nfold cross-validation number (default 10)")
    evaluation_group.add_argument("-s","--shuffle",
                                  default=False, action="store_true",
                                  help="if set, ensure a different cross-validation of files is done, otherwise, the same file splitting is done everytime")
    evaluation_group.add_argument("--correction", "-c",
                                  default=1.0, type=float,
                                  help="if input is already a restriction on the full task, this options defines a correction to apply on the final recall score to have the real scores on the full corpus")
    evaluation_group.add_argument("--unlabelled", "-u",
                                  default=False, action="store_true",
                                  help="force unlabelled evaluation, even if the prediction is made with relations")
    evaluation_group.add_argument("--accuracy", "-a",
                                  default=False, action="store_true",
                                  help="provide accuracy scores for classifiers used")
    # simple parser with separate train/test
    # todo for options
    # RFC type
    # beam size
    # nbest eval (when implemented)
    #

    args = parser.parse_args()


    features = args_to_features(args)

    data_attach = Orange.data.Table(args.data_attach)
    # print "DATA ATTACH:", data_attach

    if args.data_relations:
        data_relations = Orange.data.Table(args.data_relations)
        with_relations = True
    else:
        data_relations = None
        with_relations = False
        labels = None

    args.relations = ["attach","relations"][with_relations]
    args.context = "window5" if "window" in args.data_attach else "full"
    args.relnb = args.data_relations.split(".")[-2][-6:] if with_relations else "-"

    RECALL_CORRECTION = args.correction

    if args.save_models or args.test_only:# training only or testing only => no folds
        args.nfold = 1

    fold_struct, selection = prepare_folds(features, args.nfold, data_attach,
                                           shuffle=args.shuffle)

    all_decoders = args_to_decoders(args)
    all_learners = args_to_learners(all_decoders[0], features, args)
    # only one learner+decoder for now
    learner = all_learners[0]
    decoder = all_decoders[0]

    # eval procedures
    output_folder = args.output
    save_results  = args.output is not None
    score_labels  = with_relations and not args.unlabelled

    learner_config = LearnerConfig(use_prob = args.use_prob)

    #
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
            model = learn_attachments(learner_config, learner, train_data_attach)

            if args.save_models:# training only
                save_model("attach.model",model)

        else:# test-only
            if args.attachment_model is None:
                sys.exit("ERROR: [test mode] attachment model must be provided with -A")
            model = load_model(args.attachment_model)

        threshold = args_to_threshold(model, decoder, requested=args.threshold)

        # test
        if not(args.save_models):# else would be training only
            test_data_attach = data_attach.select_ref(selection, test_fold)
        #
        if with_relations  and not(args.test_only):
            if args.save_models:
                train_data_relations = data_relations.select_ref(selection, test_fold)
            else:
                train_data_relations = data_relations.select_ref(selection, test_fold, negate = 1)
            train_data_relations = related_relations(features, train_data_relations)
            # train model
            model_relations = learn_relations(learner_config, learner, train_data_relations)
            if args.save_models:# training only
                save_model("relations.model",model_relations)
        elif with_relations and not(args.save_models):
            test_data_relations = data_relations.select_ref(selection, test_fold)
            if args.test_only:
                model_relations = load_model(args.relation_model)
        else:# no relations
            model_relations = None

        if args.save_models:# training done, leaving
            print >> sys.stderr, "done with training, exiting"
            sys.exit(0)

        # decoding options for this fold
        config = DecoderConfig(features=features,
                               decoder=decoder,
                               model_attach=model,
                               model_relations=model_relations,
                               threshold=threshold,
                               post_labelling=args.post_label,
                               use_prob=args.use_prob)
        # -- file level --
        fold_evals = []
        for onedoc in fold_struct:
            if fold_struct[onedoc] == test_fold:
                print >> sys.stderr, "decoding on file : ", onedoc

                attach_instances  = entries_in_grouping(features, onedoc, data_attach)
                if data_relations:
                    rel_instances = entries_in_grouping(features, onedoc, data_relations)
                else:
                    rel_instances = None

                predicted = decode_document(config, model, attach_instances, rel_instances)

                if save_results:
                    exportGraph(predicted, onedoc, output_folder)

                if not args.test_only:
                    reference = related_attachments(features, attach_instances)
                    labels = related_relations(features, rel_instances) if score_labels else None
                    scores = discourse_eval(predicted, reference, labels = labels)
                    evals.append(scores)
                    fold_evals.append(scores)

        if not(args.test_only):
            fold_report = Report(fold_evals, params = args, correction = RECALL_CORRECTION)
            print "Fold eval:", fold_report.summary()
        # --end of file level
       # --- end of fold level
    # end of test for a set of parameter
    # report: summing : TODO: must register many runs with change of parameters
    if not(args.test_only):
        report = Report(evals, params = args, correction = RECALL_CORRECTION)
        print ">>> FINAL EVAL:", report.summary()
        report.save("results/"+"{relations}_{context}_{relnb}_{decoders}_{learners}_{heuristics}_{unlabelled}_{post_label}_{rfc}".format(**args.__dict__))

if __name__ == "__main__":
    main()
