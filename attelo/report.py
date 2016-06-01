"""
Experiment results
"""

from __future__ import print_function
from collections import namedtuple
import itertools
import six
import sys

from tabulate import tabulate

from .score import (Count, CSpanCount, EduCount)
from .util import (concat_l)

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

ScoreConfig = namedtuple("ScoreConfig", "correction prefix")

DEFAULT_SCORE_CONFIG = ScoreConfig(correction=1.0,
                                   prefix=None)


# pylint: disable=invalid-name
class Score(object):
    """
    Tuple of precision, recall and F1, with correction.

    Parameters
    ----------
    precision : float

    recall : float

    correction : float
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

    @classmethod
    def score(cls, tpos, tpos_fpos, tpos_fneg, correction):
        """
        From counts to precision/recall scores
        """
        return cls(_sloppy_div(tpos, tpos_fpos),
                   _sloppy_div(tpos, tpos_fneg),
                   correction)

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

        res = ["p" if config.prefix is None else "%s p" % config.prefix,
               "r", "f1"]

        if corrected:
            res += ["r (cr %.2f)" % config.correction,
                    "f1 (cr %.2f)" % config.correction]

        return res


class EduScore(object):
    """
    Accuracy scores
    """
    def __init__(self, accuracy):
        self.accuracy = accuracy

    @classmethod
    def score(cls, correct, total):
        """
        Compute the attachment precision, recall, etc based on this count
        """
        return cls(_sloppy_div(correct, total))

    @classmethod
    def score_attach(cls, count):
        """
        Compute the attachment accurracy
        """
        return cls.score(count.correct_attach, count.total)

    @classmethod
    def score_label(cls, count):
        """
        Compute the labeling accuracy
        """
        return cls.score(count.correct_label, count.total)

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
        """
        Parameters
        ----------
        evals : list of pair of Count
            Undirected and directed Count.
        """
        evals_ = ([e[0] for e in evals], [e[1] for e in evals])
        totals = (Count.sum(evals_[0]), Count.sum(evals_[1]))

        self.config = ScoreConfig(prefix=None,
                                  correction=correction)
        self.attach_undir =\
            Multiscore.create(lambda x: Score.score(x.tpos_attach,
                                                    x.tpos_fpos,
                                                    x.tpos_fneg,
                                                    correction),
                              totals[0], evals_[0])
        self.attach_dir =\
            Multiscore.create(lambda x: Score.score(x.tpos_attach,
                                                    x.tpos_fpos,
                                                    x.tpos_fneg,
                                                    correction),
                              totals[1], evals_[1])
        self.label =\
            Multiscore.create(lambda x: Score.score(x.tpos_label,
                                                    x.tpos_fpos,
                                                    x.tpos_fneg,
                                                    correction),
                              totals[1], evals_[1])
        self.params = params if params is not None else {}

    def for_json(self):
        """
        Return a JSON-serialisable dictionary representing the scores
        for this run
        """
        return {"attachment_undirected": self.attach_undir.for_json(),
                "attachment_directed": self.attach_dir.for_json(),
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
                  "\tATT", self.attach_dir.summary(),
                  "\t+DIR", self.attach_undir.summary(),
                  "\t+LAB", self.label.summary()]
        return " ".join(output)

    def table_row(self):
        "Scores as a tabulate table row"
        return (self.attach_undir.table_row() +
                self.attach_dir.table_row() +
                self.label.table_row())

    @classmethod
    def table_header(cls, config=None):
        "Scores as a tabulate table row"
        config = config or DEFAULT_SCORE_CONFIG
        config_a = ScoreConfig(correction=config.correction,
                               prefix="ATT")
        config_d = ScoreConfig(correction=config.correction,
                               prefix="+DIR")
        config_l = ScoreConfig(correction=config.correction,
                               prefix="+LAB")
        return (Multiscore.table_header(config_a) +
                Multiscore.table_header(config_d) +
                Multiscore.table_header(config_l))


class CSpanReport(object):
    """
    Experimental results and some basic statistical tests on them

    Notes
    -----
    TODO TODO TODO TODO
    """
    def __init__(self, evals, params=None, correction=1.0):
        evals_ = ([e[0] for e in evals],
                  [e[1] for e in evals],
                  [e[2] for e in evals],
                  [e[3] for e in evals])
        totals = (CSpanCount.sum(evals_[0]),
                  CSpanCount.sum(evals_[1]),
                  CSpanCount.sum(evals_[2]),
                  CSpanCount.sum(evals_[3]))
        self.config = ScoreConfig(prefix=None,
                                  correction=correction)
        # scores
        self.score_s = Multiscore.create(
            lambda x: Score.score(x.tpos, x.tpos_fpos, x.tpos_fneg,
                                  correction),
            totals[0], evals_[0])
        self.score_sn = Multiscore.create(
            lambda x: Score.score(x.tpos, x.tpos_fpos, x.tpos_fneg,
                                  correction),
            totals[1], evals_[1])
        self.score_sr = Multiscore.create(
            lambda x: Score.score(x.tpos, x.tpos_fpos, x.tpos_fneg,
                                  correction),
            totals[2], evals_[2])
        self.score_snr = Multiscore.create(
            lambda x: Score.score(x.tpos, x.tpos_fpos, x.tpos_fneg,
                                  correction),
            totals[3], evals_[3])

        self.params = params if params is not None else {}

    def for_json(self):
        """
        Return a JSON-serialisable dictionary representing the scores
        for this run
        """
        return {
            "span": self.score_s.for_json(),
            "span+nuclearity": self.score_sn.for_json(),
            "span+relation": self.score_sr.for_json(),
            "span+nuclearity+relation": self.score_snr.for_json(),
        }

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

        output = [
            self._params_to_filename(),
            "\tS", self.score_s.summary(),
            "\tS+N", self.score_sn.summary(),
            "\tS+R", self.score_sr.summary(),
            "\tS+N+R", self.score_snr.summary(),
        ]
        return " ".join(output)

    def table_row(self):
        "Scores as a tabulate table row"
        return (self.score_s.table_row() +
                self.score_sn.table_row() +
                self.score_sr.table_row() +
                self.score_snr.table_row())

    @classmethod
    def table_header(cls, config=None):
        "Scores as a tabulate table row"
        config = config or DEFAULT_SCORE_CONFIG
        config_s = ScoreConfig(correction=config.correction,
                               prefix="S")
        config_sn = ScoreConfig(correction=config.correction,
                                prefix="S+N")
        config_sr = ScoreConfig(correction=config.correction,
                                prefix="S+R")
        config_snr = ScoreConfig(correction=config.correction,
                                 prefix="S+N+R")
        return (Multiscore.table_header(config_s) +
                Multiscore.table_header(config_sn) +
                Multiscore.table_header(config_sr) +
                Multiscore.table_header(config_snr))


class LabelReport(object):
    """
    Weight and pr/rec/f1 on labels

    No distinction between attach/label needed because it's
    the same
    """
    def __init__(self, evals, correction=1.0):
        evals_ = ([e[0] for e in evals], [e[1] for e in evals])
        totals = (Count.sum(evals_[0]), Count.sum(evals_[1]))
        self.score = Score.score(totals[1].tpos_attach, totals[1].tpos_fpos,
                                 totals[1].tpos_fneg, correction)
        self.count = totals[1].tpos_fneg

    def for_json(self):
        """
        Return a JSON-serialisable dictionary representing the scores
        for this run
        """
        return {"score": self.score.for_json(),
                "count": self.count}

    def table_row(self):
        "Scores as a tabulate table row"
        return [self.count] + self.score.table_row()

    @classmethod
    def table_header(cls, config=None):
        "Scores as a tabulate table row"
        config = config or DEFAULT_SCORE_CONFIG
        config_l = ScoreConfig(correction=config.correction,
                               prefix="ATT/LAB")
        return ["count"] + Score.table_header(config_l)


class EduReport(object):
    """
    EDU-level results
    """
    def __init__(self):
        self.totals = EduCount(0, 0, 0)
        self.attach = EduScore(0)
        self.label = EduScore(0)

    def add(self, new_eval):
        "add new EDU level counts to current total"
        self.totals += new_eval
        self.attach = EduScore.score_attach(self.totals)
        self.label = EduScore.score_label(self.totals)

    def for_json(self):
        """
        Return a JSON-serialisable dictionary representing the scores
        for this run
        """
        return {"attachment": self.attach.for_json(),
                "labeling": self.label.for_json()}

    def table_row(self):
        "Scores as a tabulate table row"
        return (self.attach.table_row() +
                self.label.table_row())

    @classmethod
    def table_header(cls, config=None):
        "Scores as a tabulate table row"
        config = config or DEFAULT_SCORE_CONFIG
        config_a = ScoreConfig(correction=config.correction,
                               prefix="ATTACH")
        config_l = ScoreConfig(correction=config.correction,
                               prefix="LABEL")
        return (EduScore.table_header(config_a) +
                EduScore.table_header(config_l))


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

    def table(self, main_header=None, sortkey=None):
        """
        2D tabular output

        :type sortkey: k, v -> a
        """
        if sortkey is None:
            keys = sorted(self.reports.keys())
        else:
            keys = [k for k, _ in
                    sorted(self.reports.items(), key=sortkey)]

        if not keys:
            raise ValueError('report must have at least one key')

        if main_header:
            headers = [main_header] + [''] * (len(list(keys[0])) - 1)
        else:
            headers = []
        headers += self.report_cls.table_header()
        rows = [list(k) + self.reports[k].table_row()
                for k in keys]
        return tabulate(rows,
                        headers=headers,
                        floatfmt=".3f")

    def for_json(self):
        """
        JSON-friendly dict of scores.
        May contain more information than the raw table
        """
        return {k: v.for_json() for k, v in self.reports.items()}


# ---------------------------------------------------------------------
# confusion matrix
# ---------------------------------------------------------------------

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


def show_confusion_matrix(labels, matrix, main_header=None):
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
    table = tabulate(headers + body)
    if main_header:
        return main_header + '\n' + table
    else:
        return table

# ---------------------------------------------------------------------
# discriminating features
# ---------------------------------------------------------------------


def _condense_cell(old, new):
    """
    Maximise readability of the new cell given that it's sitting
    below the old one in a 2D table
    """
    if isinstance(new, six.string_types):
        is_eqish = lambda (x, y): x == y and '=' not in [x, y]
        zipped = list(itertools.izip_longest(old, new))
        prefix = itertools.takewhile(is_eqish, zipped)
        suffix = itertools.dropwhile(is_eqish, zipped)
        return ''.join(['.' for _ in prefix] +
                       [n if n is not None else '' for _, n in suffix])
    else:
        return '{:.2f}'.format(new)


def _condense_table(rows):
    """
    Make a table more readable by replacing identical columns in
    subsequent rows by "
    """
    if not rows:
        return rows
    results = []
    current_row = ['' for _ in rows[0]]
    for row in rows:
        new_row = [row[0]]
        new_row.extend(_condense_cell(old, new)
                       for old, new in zip(current_row[1:], row[1:]))
        results.append(new_row)
        current_row = row
    return results


def _sort_table(rows):
    """
    Return rows in the following order

    * UNRELATED always comes first
    * otherwise, sort by the names of top N features

    The hope is that this would visually group together the same
    features so you can see a natural separation
    """
    label_value = {'UNRELATED': -2}

    def ordering_key(row):
        "tweaked version of list of sorting"
        label = label_value.get(row[0], 0)
        rest = row[1::2]
        return (label, rest)

    return sorted(rows, key=ordering_key)


def show_discriminating_features(listing):
    """Build a table of discriminating features per label.

    Given a list of discriminating features for each label,
    return a string containing a hopefully friendly 2D table
    visualisation.

    Parameters
    ----------
    listing: list of (string, list of (string, float))
        List of (label, features) pairs; the features are themselves a
        list of (feature, weight) pairs.
    """
    rows = []
    for label, feats in listing:
        rows.append([label] + list(feats[0]))
        rows.extend([''] + list(x) for x in feats[1:])
    return tabulate(rows)
