"""
Experiment results
"""

from __future__ import print_function
from collections import namedtuple
import copy
import csv
import os
import sys
import cPickle

from tabulate import tabulate

try:
    STATS = True
    from scipy.stats import wilcoxon, ttest_rel, mannwhitneyu, sem, bayes_mvs
except:
    STATS = False
    print("no module scipy.stats, cannot test statistical "
          "significance of results",
          file=sys.stderr)


class AtteloReportException(Exception):
    def __init__(self, msg):
        super(self, AtteloReportException).__init__(msg)


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

    _FIELDNAMES = ["doc",
                   "num_correctly_attached",
                   "num_correctly_labeled",
                   "num_attached_predicted",
                   "num_attached_reference"
                   ]

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

    @classmethod
    def write_csv(cls, scores, counts_file):
        """
        Write counts out for any predictions that we made
        (counts_file is any python file object, eg. sys.stdout)

        :param scores: mapping from document to count
        :type scores: dict(string, Count)
        """
        writer = csv.writer(counts_file)
        writer.writerow(cls._FIELDNAMES)
        for doc, count in scores.items():
            writer.writerow([doc,
                             count.correct_attach,
                             count.correct_label,
                             count.total_predicted,
                             count.total_reference])

    @classmethod
    def read_csv(cls, counts_file):
        """
        Read a counts file into a dictionary mapping
        documents to Count objects
        """
        reader = csv.DictReader(counts_file, fieldnames=cls._FIELDNAMES)
        header_row = reader.next()
        header = [header_row[k] for k in cls._FIELDNAMES]
        if header != cls._FIELDNAMES:
            oops = "Malformed counts file (expected keys: %s, got: %s)"\
                % (cls._FIELDNAMES, header)
            raise AtteloReportException(oops)

        def mk_count(row):
            "row to Count object"
            return cls(*[int(row[a]) for a in cls._FIELDNAMES[1:]])

        return {r["doc"]: mk_count(r) for r in reader}

ScoreConfig = namedtuple("ScoreConfig", "correction prefix")

DEFAULT_SCORE_CONFIG = ScoreConfig(correction=1.0,
                                   prefix=None)


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
        if self.f1_corr is not None:
            res["f1_corrected"] = self.f1_corr
            res["recall_corrected"] = self.recall_corr
        return res

    def table_row(self):
        "Scores as a row of floats (meant to be included in a table)"
        row = [100 * self.precision,
               100 * self.recall,
               100 * self.f1]

        if self.f1_corr:
            row += [100 * self.recall_corr,
                    100 * self.f1_corr]

        return row

    @classmethod
    def table_header(cls, config=None):
        """
        Header for table using these scores (prefix to distinguish
        between different kinds of scores)
        """
        config = config or DEFAULT_SCORE_CONFIG
        corrected = config.correction != 1.0

        res = ["pre" if config.prefix is None else "%s pre" % config.prefix,
               "rec", "f1"]

        if corrected:
            res += ["rec (cr %.2f)" % config.correction,
                    "f1 (cr %.2f)" % config.correction]

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
        scores["standard_error"] = self.standard_error(_f1)
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
        if self.score.f1_corr is not None:
            output.append("\t with recall correction estimate, "
                          "R=%1.3f, F1=%1.3f" %
                          (self.score.recall_corr,
                           self.score.f1_corr))
        return " ".join(output)

    def table_row(self):
        "Scores as a row of floats (meant to be included in a table)"
        return self.score.table_row()

    @classmethod
    def table_header(cls, config=None):
        "Scores as a row of floats (meant to be included in a table)"
        return Score.table_header(config=config)


class Report(object):
    """
    Experimental results and some basic statistical tests on them
    """
    def __init__(self, evals, params=None, correction=1.0):
        totals = Count.sum(evals)
        self.config = ScoreConfig(prefix=None,
                                  correction=correction)
        self.attach = Multiscore.create(lambda x: x.score_attach(correction),
                                        totals, evals)
        self.label = Multiscore.create(lambda x: x.score_label(correction),
                                       totals, evals)
        self.params = params if params is not None else {}

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

        res = copy.copy(self.params.__dict__)
        del res["func"]
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

    def table_row(self):
        "Scores as a tabulate table row"
        return\
            self.attach.table_row() +\
            self.label.table_row()

    @classmethod
    def table_header(cls, config=None):
        "Scores as a tabulate table row"
        config = config or DEFAULT_SCORE_CONFIG
        config_a = ScoreConfig(correction=config.correction,
                               prefix="ATTACH")
        config_l = ScoreConfig(correction=config.correction,
                               prefix="LABEL")
        return\
            Multiscore.table_header(config_a) +\
            Multiscore.table_header(config_l)




class CombinedReport(object):
    """
    Report for many different configurations
    """
    #pylint: disable=pointless-string-statement
    def __init__(self, reports):
        self.reports = reports
        "dictionary from config name (string) to Report"
    #pylint: enable=pointless-string-statement

    def table(self):
        """
        2D tabular output
        """
        keys = sorted(self.reports.keys())
        return tabulate([[k] + self.reports[k].table_row() for k in keys],
                        headers=Report.table_header(),
                        floatfmt=".1f")

    def for_json(self):
        """
        JSON-friendly dict of scores.
        May contain more information than the raw table
        """
        return {k: v.for_json() for k, v in self.reports.items()}
