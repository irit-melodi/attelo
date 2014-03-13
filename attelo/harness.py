"""
July 2012

attachment decoding: from local prediction, optimize a discourse structure
while respecting a chosen set of constraints, such as: MST decoding,
incremental decoding with a right frontier constraint, etc

should regroup
      x- MST decoding
      x- A* decoding with various heuristics
      x- baseline: local, small beam search

example: cf coling_expes.sh



TODO:
 - abstract main as processing method, depending on various things: fold nb,
   learner, decoder, eval function for one discourse and within that, abstract
   layers : fold,document
 - more generic descriptions of features names
 - n best out of A* decoding, to prepare ranking
                x- generate all solution in generic astar
                 - add parameter to decoding+process_document ? but the whole
                   API for prediction assumes only one prediction
                 - revamp scores to allow eval in nbest list (eg: at given
                   depth and/or oracle in list; what else?)
 - other evals: tree-edit, parseval-like, other?
 x- RFC with coord-subord distinction
 - nicer report for scores (table, latex, figures)
"""

from collections import namedtuple
import argparse
import csv
import os
import sys
import cPickle

from ConfigParser import ConfigParser
from Orange.classification import Classifier
import Orange
try:
    STATS = True
    from scipy.stats import wilcoxon, ttest_rel, mannwhitneyu, sem, bayes_mvs
except:
    print >> sys.stderr,\
        "no module scipy.stats, cannot test stat. significance of results"
    STATS = False

from attelo.decoding.astar import astar_decoder, h0, h_best, h_max, h_average
from attelo.decoding.baseline import local_baseline, last_baseline
from attelo.decoding.mst import MST_list_edges as MST_decoder
from attelo.decoding.greedy import locallyGreedy
from attelo.learning.megam import MaxentLearner
from attelo.learning.perceptron import\
    PerceptronArgs, Perceptron, StructuredPerceptron
from attelo.edu import mk_edu_pairs
from attelo.features import Features
from attelo.fileNfold import make_n_fold, makeFoldByFileIndex
from attelo.report import Report

# from MST import MSTdecoder

ANNODIS_FEATURES = Features()  # default settings

# ---------------------------------------------------------------------
# utilities
# ---------------------------------------------------------------------


def related_attachments(features, table):
    """Return just the entries in the attachments table that
    represent related EDU pairs
    """
    return table.filter_ref({features.label: "True"})


def related_relations(features, table):
    """Return just the entries in the relations table that represent
    related EDU pair
    """
    return table.filter_ref({features.label: ["UNRELATED"]}, negate=1)


def _subtable_in_grouping(features, grouping, table):
    """Return the entries in the table that belong in the given
    group
    """
    return table.filter_ref({features.grouping: grouping})


def select_data_in_grouping(features, grouping, data_attach, data_relations):
    """Return only the data that belong in the given group
    """
    attach_instances = _subtable_in_grouping(features,
                                             grouping,
                                             data_attach)
    if data_relations:
        rel_instances = _subtable_in_grouping(features,
                                              grouping,
                                              data_relations)
    else:
        rel_instances = None
    return attach_instances, rel_instances

# ---------------------------------------------------------------------
#
# ---------------------------------------------------------------------


# TODO: describe model type
def load_model(filename):
    """
    Load model into memory from file
    """
    with open(filename, "rb") as f:
        return cPickle.load(f)


def save_model(filename, model):
    """
    Dump model into a file
    """
    with open(filename, "wb") as f:
        cPickle.dump(model, f)

# ---------------------------------------------------------------------
#
# ---------------------------------------------------------------------


def discourse_eval(features, predicted, reference, labels=None, debug=False):
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
        if (arg1, arg2) in dict_predicted:
            if labels is None:
                score += 1
                if debug:
                    print >> sys.stderr, "correct"
            else:
                relation_ref = labels.filter_ref({features.source: [arg1],
                                                  features.target: [arg2]})
                if len(relation_ref) == 0:
                    print >> sys.stderr,\
                        "attached pair without corresponding relation",\
                        one[features.grouping], arg1, arg2
                else:
                    relation_ref = relation_ref[0][features.label].value
                    score += (dict_predicted[(arg1, arg2)] == relation_ref)

    total_ref = len(reference)
    total_pred = len(predicted)
    return score, total_pred, total_ref


def combine_probs(features,
                  attach_instances,
                  rel_instances,
                  attachmt_model,
                  relations_model):
    """retrieve probability of the best relation on an edu pair, given the
    probability of an attachment
    """
    # !! instances set must correspond to same edu pair in the same order !!
    distrib = []

    edu_pair = mk_edu_pairs(features, attach_instances.domain)
    attach_instances = sorted(attach_instances, key=lambda x: x.get_metas())
    rel_instances = sorted(rel_instances, key=lambda x: x.get_metas())

    inst_pairs = zip(attach_instances, rel_instances)
    for i, (attach, relation) in enumerate(inst_pairs):
        p_attach = attachmt_model(attach, Classifier.GetProbabilities)[1]
        p_relations = relations_model(relation, Classifier.GetBoth)
        if not instance_check(features, attach, relation):
            print >> sys.stderr,\
                "mismatch of attachment/relation instance, instance number",\
                i,\
                meta_info(attach,   features),\
                meta_info(relation, features)
        # this should be investigated
        try:
            best_rel = p_relations[0].value
        except:
            best_rel = p_relations[0]

        rel_prob = max(p_relations[1])
        edu1, edu2 = edu_pair(attach)
        distrib.append((edu1, edu2, p_attach * rel_prob, best_rel))
    return distrib


def index_by_metas(instances, metas=None):
    """transform a data table to a dictionary of instances indexed by ordered
    tuple of all meta-attributes; convenient to find instances associated to
    multiple tables (eg edu pairs for attachment+relations)
    """
    if metas is None:
        to_keep = lambda x: x.get_metas().values()
    else:
        to_keep = lambda x: [x[y] for y in metas]
    result = [(tuple([y.value for y in to_keep(x)]), x) for x in instances]
    return dict(result)


def add_labels(features, predicted, rel_instances, relations_model):
    """ predict labels for a given set of edges (=post-labelling an unlabelled
    decoding)
    """
    rels = index_by_metas(rel_instances,
                          metas=[features.source, features.target])
    result = []
    for (a1, a2, _r) in predicted:
        instance_rel = rels[(a1, a2)]
        rel = relations_model(instance_rel,
                              Classifier.GetValue)
        result.append((a1, a2, rel))
    return result


def instance_check(features, one, two):
    """
    Return True if the two annotations should be considered as refering to the
    same EDU pair. This can be used as a sanity check when zipping two datasets
    that are expected to be on the same EDU pairs.
    """
    return\
        one[features.source] == two[features.source] and\
        one[features.target] == two[features.target] and\
        one[features.grouping] == two[features.grouping]


def meta_info(features, instance):
    return "%s: %s-%s" % (instance[features.grouping],
                          instance[features.source],
                          instance[features.target])


def exportGraph(predicted, doc, folder):
    fname = os.path.join(folder, doc + ".rel")
    if not os.path.exists(folder):
        os.makedirs(folder)
    f = open(fname, 'w')
    for (a1, a2, rel) in predicted:
        f.write(rel + " ( " + a1 + " / " + a2 + " )\n")
    f.close()


def export_csv(features, predicted, doc, attach_instances, folder):
    fname = os.path.join(folder, doc + ".csv")
    if not os.path.exists(folder):
        os.makedirs(folder)
    predicted_map = {(e1, e2): label for e1, e2, label in predicted}
    metas = attach_instances.domain.getmetas().values()

    with open(fname, 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(["m#" + x.name for x in metas] +
                        ["c#" + features.label])
        for r in attach_instances:
            row = [r[x].value for x in metas]
            e1 = r[features.source].value
            e2 = r[features.target].value
            epair = (e1, e2)
            label = predicted_map.get((e1, e2), "UNRELATED")
            writer.writerow(row + [label])

# ---------------------------------------------------------------------
# processing a single document
# ---------------------------------------------------------------------


# TODO: replace with named tuple
class DecoderConfig(object):
    def __init__(self, features, decoder,
                 threshold=None,
                 post_labelling=False,
                 use_prob=True):
        self.features = features
        self.decoder = decoder
        self.threshold = threshold
        self.post_labelling = post_labelling
        self.use_prob = use_prob


def decode_document(config,
                    model_attach, attach_instances,
                    model_relations=None, rel_instances=None):
    """
    decode one document (onedoc), selecting instances for attachment from
    data_attach, (idem relations if present), using trained model, model

    Return the predictions made

    TODO: check that call to learner can be uniform with 2 parameters (as
    logistic), as the documentation is inconsistent on this
    """
    features = config.features
    decoder = config.decoder
    threshold = config.threshold
    use_prob = config.use_prob

    if rel_instances and not config.post_labelling:
        prob_distrib = combine_probs(features,
                                     attach_instances, rel_instances,
                                     model_attach, model_relations)
    elif model_attach.name in ["Perceptron", "StructuredPerceptron"]:
        # home-made online models
        prob_distrib = model_attach.get_scores(attach_instances,
                                               use_prob=use_prob)
    else:
        # orange-based models
        edu_pair = mk_edu_pairs(features, attach_instances.domain)
        prob_distrib = []
        for one in attach_instances:
            edu1, edu2 = edu_pair(one)
            probs = model_attach(one, Classifier.GetProbabilities)[1]
            prob_distrib.append((edu1, edu2, probs, "unlabelled"))
    # print prob_distrib

    # get prediction (input is just prob_distrib)
    if threshold:
        predicted = decoder(prob_distrib,
                            threshold=threshold,
                            use_prob=use_prob)
        # predicted = decoder(prob_distrib, threshold = threshold)
    else:
        predicted = decoder(prob_distrib,
                            use_prob=use_prob)
        # predicted = decoder(prob_distrib)

    if config.post_labelling:
        predicted = add_labels(features,
                               predicted,
                               rel_instances,
                               model_relations)
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

    fold_struct = make_n_fold(table,
                              folds=num_folds,
                              meta_index=features.grouping)
    selection = makeFoldByFileIndex(table,
                                    fold_struct,
                                    meta_index=features.grouping)
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
    else:
        # annodis config as default, should not cause regression on coling
        # experiment
        return ANNODIS_FEATURES


def args_to_data(args):
    """
    Given the (parsed) command line arguments, return the data in
    table form
    """
    data_attach = Orange.data.Table(args.data_attach)
    if args.data_relations:
        data_relations = Orange.data.Table(args.data_relations)
    else:
        data_relations = None
    return data_attach, data_relations


def _mk_astar_decoder(heuristics, rfc):
    """
    Return an A* decoder using the given heuristics and
    right frontier constraint parameter
    """
    return lambda x, **kargs:\
        astar_decoder(x, heuristics=heuristics, RFC=rfc, **kargs)


def _known_heuristics():
    """
    Return a dictionary of possible A* heuristics.
    This lets us grab at the names of known heuristics
    for command line restruction
    """
    return {"average": h_average,
            "best": h_best,
            "max": h_max,
            "zero": h0}


def _known_decoders(heuristics, rfc):
    """
    Return a dictionary of possible decoders.
    This lets us grab at the names of known decoders
    """
    return {"last": last_baseline,
            "local": local_baseline,
            "locallyGreedy": locallyGreedy,
            "mst": MST_decoder,
            "astar": _mk_astar_decoder(heuristics, rfc)}


def _known_learners(decoder, features, perc_args):
    """
    Given the (parsed) command line arguments, return a sequence of
    learners in the order they were requested on the command line
    """

    # orange classifiers
    bayes = Orange.classification.bayes.NaiveLearner(adjust_threshold=True)
    bayes.name = "naive bayes"
    #svm = Orange.classification.svm.SVMLearnerEasy(probability = True)
    svm = Orange.classification.svm.SVMLearner(probability=True)
    svm.name = "svm"
    maxent = MaxentLearner()  # Orange.classification.logreg.LogRegLearner()
    maxent.name = "maxent"
    majority = Orange.classification.majority.MajorityLearner()
    majority.name = "majority"

    # home made perceptron
    perc = Perceptron(features=features,
                      nber_it=perc_args.iterations,
                      avg=perc_args.averaging)
    # home made structured perceptron
    struc_perc = StructuredPerceptron(features, decoder,
                                      nber_it=perc_args.iterations,
                                      avg=perc_args.averaging,
                                      use_prob=perc_args.use_prob)

    return {"bayes": bayes,
            "svm": svm,
            "maxent": maxent,
            "majority": majority,
            "perc": perc,
            "struc_perc": struc_perc}


# these are just dummy values (we just want the keys here)
KNOWN_HEURISTICS = _known_heuristics().keys()
KNOWN_DECODERS = _known_decoders([], False).keys()
KNOWN_LEARNERS = _known_learners(last_baseline, {},
                                 PerceptronArgs(0, False, False)).keys()

def args_to_decoders(args):
    """
    Given the (parsed) command line arguments, return a sequence of
    decoders in the order they were requested on the command line
    """
    if args.heuristics not in _known_heuristics():
        raise argparse.ArgumentTypeError("Unknown heuristics: %s" %
                                         args.heuristics)
    heuristic = _known_heuristics().get(args.heuristics, h_average)
    if not args.data_relations:
        args.rfc = "simple"

    _decoders = _known_decoders(heuristic, args.rfc)
    requests = args.decoders.split(",")
    unknown = ",".join(r for r in requests if r not in _decoders)
    if unknown:
        raise argparse.ArgumentTypeError("Unknown decoders: %s" % unknown)
    else:
        return [_decoders[x] for x in requests]


def args_to_learners(decoder, features, args):
    """
    Given the (parsed) command line arguments, return a sequence of
    learners in the order they were requested on the command line
    """

    perc_args = PerceptronArgs(iterations=args.nit,
                               averaging=args.averaging,
                               use_prob=args.use_prob)
    _learners = _known_learners(decoder, features, perc_args)
    requests = args.learners.split(",")
    unknown = ",".join(r for r in requests if r not in _learners)
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


def command_save_models(args):
    data_attach, data_relations = args_to_data(args)
    features = args_to_features(args)
    all_decoders = args_to_decoders(args)
    all_learners = args_to_learners(all_decoders[0], features, args)
    # only one learner+decoder for now
    learner = all_learners[0]
    decoder = all_decoders[0]

    print >> sys.stderr, ">>> training ... "
    model_attach = learner(data_attach)
    save_model("attach.model", model_attach)

    if data_relations:
        related_only = related_relations(features, data_relations)
        model_relations = learner(related_only)
        save_model("relations.model", model_relations)

    print >> sys.stderr, "done with training, exiting"
    sys.exit(0)


def command_test_only(args):
    data_attach, data_relations = args_to_data(args)
    features = args_to_features(args)
    all_decoders = args_to_decoders(args)
    # only one learner+decoder for now
    decoder = all_decoders[0]

    if not args.attachment_model:
        sys.exit("ERROR: [test mode] attachment model must be provided " +
                 "with -A")
    if data_relations and not args.relation_model:
        sys.exit("ERROR: [test mode] relation model must be provided if " +
                 "relation data is provided")

    model_attach = load_model(args.attachment_model)
    model_relations = load_model(args.relation_model) if data_relations\
        else None

    threshold = args_to_threshold(model_attach,
                                  decoder,
                                  requested=args.threshold)

    config = DecoderConfig(features=features,
                           decoder=decoder,
                           threshold=threshold,
                           post_labelling=args.post_label,
                           use_prob=args.use_prob)

    grouping_index = data_attach.domain.index(features.grouping)
    all_groupings = set()
    for inst in data_attach:
        all_groupings.add(inst[grouping_index].value)

    for onedoc in all_groupings:
        print >> sys.stderr, "decoding on file : ", onedoc

        attach_instances, rel_instances =\
            select_data_in_grouping(features,
                                    onedoc,
                                    data_attach,
                                    data_relations)

        predicted = decode_document(config,
                                    model_attach, attach_instances,
                                    model_relations, rel_instances)
        exportGraph(predicted, onedoc, args.output)
        export_csv(features, predicted, onedoc, attach_instances, args.output)


def command_nfold_eval(args):
    features = args_to_features(args)
    data_attach, data_relations = args_to_data(args)

    all_decoders = args_to_decoders(args)
    all_learners = args_to_learners(all_decoders[0], features, args)
    # only one learner+decoder for now
    learner = all_learners[0]
    decoder = all_decoders[0]

    RECALL_CORRECTION = args.correction

    fold_struct, selection = prepare_folds(features, args.nfold, data_attach,
                                           shuffle=args.shuffle)

    with_relations = bool(data_relations)
    args.relations = ["attach", "relations"][with_relations]
    args.context = "window5" if "window" in args.data_attach else "full"

    # FIXME: this appears to be an Annodis internal filename convention
    # and should be kicked out altogether
    args.relnb = args.data_relations.split(".")[-2][-6:] if with_relations\
        else "-"

    # eval procedures
    score_labels = with_relations and not args.unlabelled

    evals = []
    # --- fold level -- to be refactored
    for test_fold in range(args.nfold):
        print >> sys.stderr, ">>> doing fold ", test_fold + 1
        print >> sys.stderr, ">>> training ... "

        train_data_attach = data_attach.select_ref(selection,
                                                   test_fold,
                                                   negate=1)
        # train model
        model_attach = learner(train_data_attach)

        # test
        test_data_attach = data_attach.select_ref(selection, test_fold)

        if with_relations:
            train_data_relations = data_relations.select_ref(selection,
                                                             test_fold,
                                                             negate=1)
            train_data_relations = related_relations(features,
                                                     train_data_relations)
            # train model
            model_relations = learner(train_data_relations)
        else:  # no relations
            model_relations = None

        # decoding options for this fold
        threshold = args_to_threshold(model_attach,
                                      decoder,
                                      requested=args.threshold)
        config = DecoderConfig(features=features,
                               decoder=decoder,
                               threshold=threshold,
                               post_labelling=args.post_label,
                               use_prob=args.use_prob)

        # -- file level --
        fold_evals = []
        for onedoc in fold_struct:
            if fold_struct[onedoc] == test_fold:
                print >> sys.stderr, "decoding on file : ", onedoc

                attach_instances, rel_instances =\
                    select_data_in_grouping(features,
                                            onedoc,
                                            data_attach,
                                            data_relations)

                predicted = decode_document(config,
                                            model_attach, attach_instances,
                                            model_relations, rel_instances)

                reference = related_attachments(features, attach_instances)
                labels =\
                    related_relations(features, rel_instances) if score_labels\
                    else None
                scores = discourse_eval(features,
                                        predicted,
                                        reference,
                                        labels=labels)
                evals.append(scores)
                fold_evals.append(scores)

        fold_report = Report(fold_evals,
                             params=args,
                             correction=RECALL_CORRECTION)
        print "Fold eval:", fold_report.summary()
        # --end of file level
       # --- end of fold level
    # end of test for a set of parameter
    # report: summing : TODO: must register many runs with change of parameters
    report = Report(evals, params=args, correction=RECALL_CORRECTION)
    print ">>> FINAL EVAL:", report.summary()
    fname_fmt = "_".join("{" + x + "}" for x in
                         ["relations",
                          "context",
                          "relnb",
                          "decoders",
                          "learners",
                          "heuristics",
                          "unlabelled",
                          "post_label",
                          "rfc"])
    report.save("results/"+fname_fmt.format(**args.__dict__))


def main():
    # TODO for options
    # RFC type
    # beam size
    # nbest eval (when implemented)
    usage = "%(prog)s [options] attachment_data_file [relation_data_file]"
    parser = argparse.ArgumentParser(usage=usage)
    subparsers = parser.add_subparsers()

    common_args = argparse.ArgumentParser(add_help=False)

    common_args.add_argument("data_attach", metavar="FILE",
                             help="attachment data")
    common_args.add_argument("data_relations", metavar="FILE", nargs="?",
                             help="relations data")  # optional
    common_args.add_argument("--config", "-C", metavar="FILE",
                             default=None,
                             help="corpus specificities config file; if "
                             "absent, defaults to hard-wired annodis config")

    learner_args = argparse.ArgumentParser(add_help=False)
    learner_args.add_argument("--learners", "-l",
                              default="bayes",
                              choices=KNOWN_LEARNERS,
                              help="comma separated list of learners for "
                              "attachment [and relations]")

    # classifier prefs
    classifier_grp = learner_args.add_argument_group('classifier arguments')
    ## classifier prefs (perceptron)
    classifier_grp.add_argument("--averaging", "-m",
                                default=False, action="store_true",
                                help="averaged perceptron")
    classifier_grp.add_argument("--nit", "-i",
                                default=1, type=int,
                                help="number of iterations for "
                                "perceptron models")

    # FIXME: this perceptron arg is for both learning/decoding
    common_args.add_argument("--use_prob", "-P",
                             default=True, action="store_false",
                             help="convert perceptron scores "
                             "into probabilities")

    # decoder prefs
    decoder_args = argparse.ArgumentParser(add_help=False)
    decoder_grp = common_args.add_argument_group('decoder arguments')
    decoder_grp.add_argument("--threshold", "-t",
                             default=None, type=float,
                             help="force the classifier to use this threshold "
                             "value for attachment decisions, unless it is "
                             "trained explicitely with a threshold")
    # FIXME: decoder_grp in common_args for now as struct_perceptron seems to
    # rely on a decoder
    decoder_grp.add_argument("--decoders", "-d", default="local",
                             choices=KNOWN_DECODERS,
                             help="comma separated list of decoders for "
                             "attachment (cf also heuristics for astar)")
    decoder_grp.add_argument("--heuristics", "-e",
                             default="average",
                             choices=KNOWN_HEURISTICS,
                             help="heuristics used for astar decoding; "
                             "default=average")
    decoder_grp.add_argument("--rfc", "-r",
                             default="full",
                             choices=["full", "simple", "none"],
                             help="with astar decoding, what kind of RFC is "
                             "applied: simple of full; simple means "
                             "everything is subordinating")

    # harness prefs (shared between eval)
    decoder_args.add_argument("--post-label", "-p",
                              default=False, action="store_true",
                              help="decode only on attachment, and predict "
                              "relations afterwards")

    # learn command
    cmd_learn = subparsers.add_parser('learn',
                                      parents=[common_args, learner_args])
    cmd_learn.set_defaults(func=command_save_models)

    # decode command
    cmd_decode = subparsers.add_parser('decode',
                                       parents=[common_args, decoder_args])
    cmd_decode.add_argument("--attachment-model", "-A", default=None,
                            help="provide saved model for prediction of "
                            "attachment (only with -T option)")
    cmd_decode.add_argument("--relation-model", "-R", default=None,
                            help="provide saved model for prediction of "
                            "relations (only with -T option)")
    cmd_decode.add_argument("--output", "-o",
                            default=None,
                            required=True,
                            metavar="DIR",
                            help="save predicted structures here")
    cmd_decode.set_defaults(func=command_test_only)

    # scoring profs
    cmd_eval = subparsers.add_parser('evaluate',
                                     parents=[common_args,
                                              learner_args,
                                              decoder_args])
    cmd_eval.set_defaults(func=command_nfold_eval)
    cmd_eval.add_argument("--nfold", "-n",
                          default=10, type=int,
                          help="nfold cross-validation number (default 10)")
    cmd_eval.add_argument("-s", "--shuffle",
                          default=False, action="store_true",
                          help="if set, ensure a different cross-validation "
                          "of files is done, otherwise, the same file "
                          "splitting is done everytime")
    cmd_eval.add_argument("--correction", "-c",
                          default=1.0, type=float,
                          help="if input is already a restriction on the full "
                          "task, this options defines a correction to apply "
                          "on the final recall score to have the real scores "
                          "on the full corpus")
    cmd_eval.add_argument("--unlabelled", "-u",
                          default=False, action="store_true",
                          help="force unlabelled evaluation, even if the "
                          "prediction is made with relations")
    cmd_eval.add_argument("--accuracy", "-a",
                          default=False, action="store_true",
                          help="provide accuracy scores for classifiers used")

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
