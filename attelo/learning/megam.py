


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
