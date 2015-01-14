


"""
use nltk megam wrapper for relation prediction.

for that, wrap the wrapper for orange consumption

TODO: render this independent from class name
"""
import sys

import Orange


# nice trick from http://stackoverflow.com/a/6796752
class RedirectStdStreams(object):
    """
    utility class to help us redirect output that would otherwise
    go to the sys.stdout or sys.stderr
    """

    def __init__(self, stdout=None, stderr=None):
        self._stdout = stdout or sys.stdout
        self._stderr = stderr or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush()
        self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush()
        self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr


try:
    import nltk.classify.megam as megam
    import nltk.classify.maxent as maxent
    with RedirectStdStreams(stdout=sys.stderr):
        megam.config_megam()
    #megam.config_megam(bin="path/to/megam")
    use_megam = True
except:
    print >> sys.stderr, "ERROR: megam not found, configure it before using this"
    print >> sys.stderr, "using default nltk  maxent"
    use_megam = False


class MaxentLearner:
    """
    wrapper to nltk version of max entropy classifier
    TODO: forces to use megam, could use a parameter for other algorithms.

    return_type: whether to return the "label" (a string) or an orange "value"
    """
    def __init__(self, name = 'maxent', return_type = "label"):
        self.name = name

    def __call__(self, data, weight = None):
        conv_data = map(orange2maxent, data)
        if use_megam:
            model = maxent.MaxentClassifier.train(conv_data, algorithm = "megam")
        else:
            model = maxent.MaxentClassifier.train(conv_data)
        return MaxentClassifier(classifier = model, domain = data.domain, name=self.name)#["CLASS"].values)

# TODO: probdist will calculate prob even for classes filtered out
# (eg UNRELATED), because they are still in the domain. 
# this is safer this way, as it ensures probs are stored in orange order
# and attachment classifier assumes fixed order (should be changed!)
# not a problem for relations, though. 
class MaxentClassifier:

    def __init__(self, name="maxent",**kwds):
        self.__dict__.update(kwds)
        self.name=name

    #def __call__(self, instance, result_type):
    def __call__(self, *args):
        if len(args)==1:
            instance = args[0]
            result_type = Orange.classification.Classifier.GetValue
        else:
             instance,result_type = args
  
        test = orange2maxent(instance)[0]
        label = self.classifier.classify(test)
        v = Orange.data.Value(self.domain.classVar, label)

        prob_dist = self.classifier.prob_classify(test)

        if result_type == Orange.classification.Classifier.GetValue:
            return v#label
        elif result_type == Orange.classification.Classifier.GetProbabilities:
            return {x: prob_dist.prob(x) for x in self.domain.classVar.values}
        else:
            return v, {x: prob_dist.prob(x) for x in self.domain.classVar.values}

    #def __call__(self,instance):
    #    return self(instance,Orange.classification.Classifier.GetValue)


def orange2maxent(example):
    """convert orange example to featureset+label for maxent API"""
    featlist = [(f.variable.name, f.value) for f in example.native() if f.variable.name != "CLASS"]
    label = example.get_class().value
    return dict(featlist), label

if __name__ == "__main__":
    """usage: prog data.csv [relation]

    relation indicates that unrelated instances are to be ignored
    """
    from orngStat import McNemar as mcnemar
    from scipy.stats import chi2

    relation = False
    data = Orange.data.Table(sys.argv[1])
    if len(sys.argv)>2:
        data = data.filter_ref({"CLASS":["UNRELATED"]}, negate = 1)
        relation = True
    # testing wrapper on given data by cross-validation
    me = MaxentLearner()
    nb = Orange.classification.bayes.NaiveLearner(adjust_threshold = True)
    cv = Orange.evaluation.testing.cross_validation([me,nb], data, folds=10)
    print "maxent versus naive bayes"
    print ["accuracy = %.4f" % score for score in Orange.evaluation.scoring.CA(cv,report_se=True)]
    print "chi2 by mcnemar = ",mcnemar(cv)[1][0]
    print "p value = %.3e"% chi2(1).sf(mcnemar(cv)[1][0])
    cm = Orange.evaluation.scoring.confusion_matrices(cv)
    if not(relation):
        print "F1 for True class: %.4f" % Orange.evaluation.scoring.F1(cm[0])
        print "P/R for True class", Orange.evaluation.scoring.precision(cm[0]), Orange.evaluation.scoring.recall(cm[0])
        print "F1 for True class: %.4f" % Orange.evaluation.scoring.F1(cm[1])
        print "P/R for True class", Orange.evaluation.scoring.precision(cm[1]), Orange.evaluation.scoring.recall(cm[1])
    #print model(data[0])
    #print model(data[0], Orange.classification.Classifier.GetValue)
    #print model(data[0], Orange.classification.Classifier.GetProbabilities)
