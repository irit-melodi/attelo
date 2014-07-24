"""
Experiment results
"""

from __future__ import print_function
import copy
import os
import sys
import cPickle

try:
    STATS = True
    from scipy.stats import wilcoxon, ttest_rel, mannwhitneyu, sem, bayes_mvs
except:
    STATS = False
    print("no module scipy.stats, cannot test statistical "
          "significance of results",
          file=sys.stderr)


def _sloppy_div(num, den):
    """
    Divide by denominator unless it's zero, in which case just return 0
    """
    return (num / float(den)) if den > 0 else 0


def _f1_score(prec, recall):
    """
    F1 measure: geometric mean of a precision/recall score
    """
    return 2 * _sloppy_div(prec * recall, prec + recall)


class Count(object):
    """
    Things we would count during the scoring process
    """

    def __init__(self,
                 correct_attach, correct_label,
                 total_predicted, total_reference):
        self.correct_attach = correct_attach
        self.correct_label = correct_label
        self.total_predicted = total_predicted
        self.total_reference = total_reference

    @classmethod
    def sum(cls, counts):
        """
        Count made of the total of all counts
        """
        return cls(sum(x.correct_attach for x in counts),
                   sum(x.correct_label for x in counts),
                   sum(x.total_predicted for x in counts),
                   sum(x.total_reference for x in counts))

    def score_attach(self, correction=1.0):
        """
        Compute the attachment precision, recall, etc based on this count
        """
        return Score(_sloppy_div(self.correct_attach, self.total_predicted),
                     _sloppy_div(self.correct_attach, self.total_reference),
                     correction)

    def score_label(self, correction=1.0):
        """
        Compute the labeling precision, recall, etc based on this count
        """
        return Score(_sloppy_div(self.correct_label, self.total_predicted),
                     _sloppy_div(self.correct_label, self.total_reference),
                     correction)


# pylint: disable=too-few-public-methods, invalid-name
class Score(object):
    """
    A basic precision, recall (and F1) tuple with correction
    """
    def __init__(self, precision, recall, correction=1.0):
        self.precision = precision
        self.recall = recall
        self.f1 = _f1_score(precision, recall)
        self.correction = correction

        if correction > 1.0:
            raise ValueError("correction must be <= 1.0 (is %f)" % correction)
        elif self.correction < 1.0:
            self.recall_corr = self.recall * self.correction
            self.f1_corr = _f1_score(self.precision, self.recall_corr)
        else:
            self.f1_corr = None
            self.recall_corr = None

    def for_json(self):
        """
        Return a JSON-serialisable dictionary representing the scores
        for this run
        """
        res = {"precision": self.precision,
               "recall": self.recall,
               "f1": self.f1}
        if self.f1_corr:
            res["f1_corrected"] = self.f1_corr
            res["recall_corrected"] = self.recall_corr
        return res
# pylint: enable=too-few-public-methods, invalid-name


class Multiscore(object):
    """
    A combined score and a series of individual document level scores
    """
    def __init__(self, score, doc_scores):
        self.score = score
        self.doc_scores = doc_scores

    @classmethod
    def create(cls, fun, total, doc_level):
        "apply a scoring function to total and individual counts"
        return cls(fun(total), list(map(fun, doc_level)))

    def map_doc_scores(self, fun):
        "apply a function on our doc-level scores"
        return list(map(fun, self.doc_scores))

    def standard_error(self, fun):
        "standard error (of the mean) on measures by text"
        return sem(self.map_doc_scores(fun))

    def confidence_interval(self, fun, alpha=0.95):
        "will return mean, confidence interval"
        return bayes_mvs(self.map_doc_scores(fun), alpha)[0]

    # inspired by Apetite Evaluation class
    def significance(self, fun, other, test="wilcoxon"):
        """computes stats significance of difference between two sets
        of scores test can be paired wilcoxon, mannwhitney for indep
        samples, or paired ttest.
        """
        scores1 = self.map_doc_scores(fun)
        scores2 = other.map_doc_scores(fun)
        if isinstance(scores1[0], float) or isinstance(scores1[0], int):
            pass
        else:
            # TODO: this is suspicious
            scores1 = [x for (x, y) in scores1]
            scores2 = [x for (x, y) in scores2]

        #differences = [(x, y) for (x, y) in zip(scores1, scores2) if x != y]
        #print >> sys.stderr, differences
        #print >> sys.stderr, d2
        #print >> sys.stderr, [x for (i,x) in enumerate(d1) if x!=d2[i]]
        assert len(scores1) == len(scores1)

        results = {}
        if test == "wilcoxon" or test == "all":
            results["wilcoxon"] = wilcoxon(scores1, scores2)[1]
        if test == "ttest" or test == "all":
            results["paired ttest"] = ttest_rel(scores1, scores2)[1]
        if test == "mannwhitney" or test == "all":
            results["mannwhitney"] = mannwhitneyu(scores1, scores2)[1]
        return results

    def for_json(self):
        """
        Return a JSON-serialisable dictionary representing the scores
        for this run, along with some additional statistical tests if
        we can manage it
        """
        scores = self.score.for_json()
        _f1 = lambda x: x.f1
        try:
            mean, (int0, _) = self.confidence_interval(_f1)
            scores["confidence_mean"] = mean
            scores["confidence_interval"] = mean - int0
        except:
            print("warning: not able to compute confidence interval",
                  file=sys.stderr)
        return scores

    def summary(self):
        "One line summary string"

        _f1 = lambda x: x.f1
        try:
            mean, (int0, _) = self.confidence_interval(_f1)
        except:
            print("warning: not able to compute confidence interval",
                  file=sys.stderr)
            mean, int0 = (0, 0)

        output = []
        output.append("Prec=%1.3f, Recall=%1.3f," %
                      (self.score.precision,
                       self.score.recall))
        output.append("F1=%1.3f +/- %1.3f (%1.3f +- %1.3f)" %
                      (self.score.f1,
                       self.standard_error(_f1),
                       mean,
                       mean - int0))
        if self.score.f1_corr:
            output.append("\t with recall correction estimate, "
                          "R=%1.3f, F1=%1.3f" %
                          (self.score.recall_corr,
                           self.score.f1_corr))
        return " ".join(output)


class Report(object):
    """
    Experimental results and some basic statistical tests on them
    """
    def __init__(self, evals, params=None, correction=1.0):
        totals = Count.sum(evals)
        self.attach = Multiscore.create(lambda x: x.score_attach(correction),
                                        totals, evals)
        self.label = Multiscore.create(lambda x: x.score_label(correction),
                                       totals, evals)
        self.params = params

    def save(self, name):
        "Dump the scores a file of the given name in some binary format"
        dname = os.path.dirname(name)
        if not os.path.exists(dname):
            os.makedirs(dname)
        with open(name, "wb") as reportfile:
            cPickle.dump(self, reportfile)

    def for_json(self):
        """
        Return a JSON-serialisable dictionary representing the scores
        for this run
        """
        attach = self.attach.for_json()
        label = self.label.for_json()
        scores = {"attachment": attach,
                  "labeling": label}
        return scores

    def json_params(self):
        """
        Return a JSON-serialisable dictionary representing the params
        for this run
        """

        res = {}
        res["params"] = copy.copy(self.params.__dict__)
        del res["params"]["func"]
        return res

    def _params_to_filename(self):
        "One line parameter listing"

        pieces = "{relations} {context} : \t "\
            "{decoder}+{learner}+{relation_learner}, "\
            "h={heuristics}, "\
            "unlabelled={unlabelled},"\
            "post={post_label},"\
            "rfc={rfc}"
        return pieces.format(**self.params.__dict__)

    def summary(self):
        "One line summary string"

        output = [self._params_to_filename(),
                  "\tATTACHMENT", self.attach.summary(),
                  "\tLABELLING", self.label.summary()]
        return " ".join(output)
