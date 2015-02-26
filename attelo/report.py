"""
Experiment results
"""

from __future__ import print_function
from collections import namedtuple
import itertools
import sys

from tabulate import tabulate

# pylint: disable=too-few-public-methods

try:
    STATS = True
    from scipy.stats import wilcoxon, ttest_rel, mannwhitneyu, sem, bayes_mvs
except ImportError:
    STATS = False
    print("no module scipy.stats, cannot test statistical "
          "significance of results",
          file=sys.stderr)


class AtteloReportException(Exception):
    '''things that arise when trying to build reports'''
    def __init__(self, msg):
        super(AtteloReportException, self).__init__(msg)


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


class Count(namedtuple('Count',
                       ['tpos_attach',
                        'tpos_label',
                        'tpos_fpos',
                        'tpos_fneg'])):
    """
    Things we would count during the scoring process
    """
    @classmethod
    def sum(cls, counts):
        """
        Count made of the total of all counts
        """
        return cls(sum(x.tpos_attach for x in counts),
                   sum(x.tpos_label for x in counts),
                   sum(x.tpos_fpos for x in counts),
                   sum(x.tpos_fneg for x in counts))

    def score_attach(self, correction=1.0):
        """
        Compute the attachment precision, recall, etc based on this count
        """
        return Score(_sloppy_div(self.tpos_attach, self.tpos_fpos),
                     _sloppy_div(self.tpos_attach, self.tpos_fneg),
                     correction)

    def score_label(self, correction=1.0):
        """
        Compute the labeling precision, recall, etc based on this count
        """
        return Score(_sloppy_div(self.tpos_label, self.tpos_fpos),
                     _sloppy_div(self.tpos_label, self.tpos_fneg),
                     correction)


class EduCount(namedtuple('EduCount',
                          ['correct_attach',
                           'correct_label',
                           'total'])):
    """
    Things we would count during the scoring process
    """
    @classmethod
    def sum(cls, counts):
        """
        Count made of the total of all counts
        """
        return cls(sum(x.correct_attach for x in counts),
                   sum(x.correct_label for x in counts),
                   sum(x.total for x in counts))

    def __add__(self, other):
        return EduCount(self.correct_attach + other.correct_attach,
                        self.correct_label + other.correct_label,
                        self.total + other.total)

    def score_attach(self):
        """
        Compute the attachment precision, recall, etc based on this count
        """
        return EduScore(_sloppy_div(self.correct_attach, self.total))

    def score_label(self):
        """
        Compute the labeling precision, recall, etc based on this count
        """
        return EduScore(_sloppy_div(self.correct_label, self.total))


ScoreConfig = namedtuple("ScoreConfig", "correction prefix")

DEFAULT_SCORE_CONFIG = ScoreConfig(correction=1.0,
                                   prefix=None)


# pylint: disable=invalid-name
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
        row = [self.precision,
               self.recall,
               self.f1]

        if self.f1_corr:
            row += [self.recall_corr,
                    self.f1_corr]

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


class EduScore(object):
    """
    Accuracy scores
    """
    def __init__(self, accuracy):
        self.accuracy = accuracy

    def for_json(self):
        """
        Return a JSON-serialisable dictionary representing the scores
        for this run
        """
        return {"accuracy": self.accuracy}

    def table_row(self):
        "Scores as a row of floats (meant to be included in a table)"
        return [self.accuracy]

    @classmethod
    def table_header(cls, config=None):
        """
        Header for table using these scores (prefix to distinguish
        between different kinds of scores)
        """
        config = config or DEFAULT_SCORE_CONFIG

        res = ["acc" if config.prefix is None else "%s acc" % config.prefix]
        return res
# pylint: enable=invalid-name


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
        return cls(fun(total), [fun(x) for x in doc_level])

    def map_doc_scores(self, fun):
        "apply a function on our doc-level scores"
        return [fun(x) for x in self.doc_scores]

    def standard_error(self, fun):
        "standard error (of the mean) on measures by text"
        return sem(self.map_doc_scores(fun))

    def confidence_interval(self, fun, alpha=0.95):
        "will return mean, confidence interval"
        return bayes_mvs(self.map_doc_scores(fun), alpha)[0]

    def _check_can_compute_confidence(self):
        '''Return 'True' if we should be able to compute a
        confidence interval; emit a warning otherwise'''

        if len(self.doc_scores) < 2:
            print("WARNING: don't have the at least data points",
                  "needed to compute compute confidence interval",
                  file=sys.stderr)
            return False
        else:
            return True

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
            scores1 = [x for x, _ in scores1]
            scores2 = [x for x, _ in scores2]

        # differences = [(x, y) for (x, y) in zip(scores1, scores2) if x != y]
        # print(difference, file=sys.stderr)
        # print(d2, file=sys.stderr)
        # print([x for (i,x) in enumerate(d1) if x!=d2[i]], file=sys.stderr)
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
        if self._check_can_compute_confidence():
            mean, (int0, _) = self.confidence_interval(_f1)
            scores["confidence_mean"] = mean
            scores["confidence_interval"] = mean - int0
        return scores

    def summary(self):
        "One line summary string"

        _f1 = lambda x: x.f1
        if self._check_can_compute_confidence():
            mean, (int0, _) = self.confidence_interval(_f1)
        else:
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


class EdgeReport(object):
    """
    Experimental results and some basic statistical tests on them
    """
    def __init__(self, evals, params=None, correction=1.0):
        totals = Count.sum(evals)
        self.config = ScoreConfig(prefix=None,
                                  correction=correction)
        self.attach =\
            Multiscore.create(lambda x: x.score_attach(correction),
                              totals, evals)
        self.label =\
            Multiscore.create(lambda x: x.score_label(correction),
                              totals, evals)
        self.params = params if params is not None else {}

    def for_json(self):
        """
        Return a JSON-serialisable dictionary representing the scores
        for this run
        """
        return {"attachment": self.attach.for_json(),
                "labeling": self.label.for_json()}

    def _params_to_filename(self):
        "One line parameter listing"

        pieces = ("{decoder}+{learner}+{relation_learner}, "
                  "h={heuristics}, "
                  "unlabelled={unlabelled},"
                  "post={post_label},"
                  "rfc={rfc}")
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


class EduReport(object):
    """
    EDU-level results
    """
    def __init__(self):
        self.totals = EduCount(0, 0, 0)
        self.attach = self.totals.score_attach()
        self.label = self.totals.score_label()

    def add(self, new_eval):
        "add new EDU level counts to current total"
        self.totals += new_eval
        self.attach = self.totals.score_attach()
        self.label = self.totals.score_label()

    def for_json(self):
        """
        Return a JSON-serialisable dictionary representing the scores
        for this run
        """
        return {"attachment": self.attach.for_json(),
                "labeling": self.label.for_json()}

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
            EduScore.table_header(config_a) +\
            EduScore.table_header(config_l)


class CombinedReport(object):
    """
    Report for many different configurations
    """
    # pylint: disable=pointless-string-statement
    def __init__(self, report_cls, reports):
        self.report_cls = report_cls
        self.reports = reports
        "dictionary from config name (string) to Report"
    # pylint: enable=pointless-string-statement

    def table(self):
        """
        2D tabular output
        """
        keys = sorted(self.reports.keys())
        rows = [list(k) + self.reports[k].table_row()
                for k in keys]
        return tabulate(rows,
                        headers=self.report_cls.table_header(),
                        floatfmt=".3f")

    def for_json(self):
        """
        JSON-friendly dict of scores.
        May contain more information than the raw table
        """
        return {k: v.for_json() for k, v in self.reports.items()}


def _mk_confusion_row(ignore, row):
    '''
    Given a list of numbers, replace the zeros by '.'.
    Put angle brackets around the column we should ignore
    '''
    res = []
    for i, col in enumerate(row):
        cell = col or '.'
        cell = '<{}>'.format(cell) if i == ignore else cell
        res.append(cell)
    return res


def show_confusion_matrix(labels, matrix):
    '''
    Return a string representing a confusion matrix in 2D
    '''
    longest_label = max(labels, key=len)
    len_longest = len(longest_label)
    rlabels = [x.rjust(len_longest, ' ') + '-' for x in labels]
    # pylint: disable=star-args
    # fake vertical headers by making them rows
    headers = [[''] + list(row)
               for row in itertools.izip_longest(*rlabels, fillvalue='')]
    # pylint: enable=star-args
    body = []
    for rnum, (label, row) in enumerate(zip(labels, matrix.tolist())):
        body.append([label] + _mk_confusion_row(rnum, row))
    return tabulate(headers + body)
