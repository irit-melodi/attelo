"""
Helpers for result reporting
"""

# Author: Eric Kow
# License: CeCILL-B (French BSD3-like)

# pylint: disable=too-few-public-methods

from __future__ import print_function
from collections import (namedtuple, defaultdict)
from os import path as fp

from .util import makedirs
from ..fold import (select_testing)
from ..report import (EdgeReport,
                      LabelReport,
                      EduReport,
                      CombinedReport,
                      show_confusion_matrix)
from ..score import (empty_confusion_matrix,
                     build_confusion_matrix,
                     score_edges,
                     score_edges_by_label,
                     score_edus)
from ..table import (DataPack)


class ReportPack(namedtuple('ReportPack',
                            ['edge',
                             'edge_by_label',
                             'edu',
                             'confusion',
                             'confusion_labels'])):
    """
    Collection of reports of various sorts

    Missing reports can be set to None (but confusion_labels
    should be set if confusion is set)

    :type edge_by_label: dict(string, CombinedReport)
    :type confusion: dict(string, array)
    """

    def dump(self, output_dir):
        """
        Save reports to an output directory
        """
        makedirs(output_dir)
        if self.edge is not None:
            # edgewise scores
            ofilename = fp.join(output_dir, 'scores.txt')
            with open(ofilename, 'w') as ostream:
                print(self.edge.table(), file=ostream)

        if self.edu is not None:
            # edu scores
            ofilename = fp.join(output_dir, 'edu-scores.txt')
            with open(ofilename, 'w') as ostream:
                print(self.edu.table(), file=ostream)

        if self.edge_by_label is not None:
            label_dir = fp.join(output_dir, 'label-scores')
            makedirs(label_dir)
            for key, report in self.edge_by_label.items():
                ofilename = fp.join(label_dir, '-'.join(key))
                with open(ofilename, 'w') as ostream:
                    print(report.table(sortkey=lambda (_, v): 0 - v.count),
                          file=ostream)

        if self.confusion is not None:
            confusion_dir = fp.join(output_dir, 'confusion')
            makedirs(confusion_dir)
            for key, matrix in self.confusion.items():
                ofilename = fp.join(confusion_dir, '-'.join(key))
                with open(ofilename, 'w') as ostream:
                    print(show_confusion_matrix(self.confusion_labels, matrix),
                          file=ostream)


class Slice(namedtuple('Slice',
                       ['fold',
                        'configuration',
                        'predictions',
                        'enable_details'])):
    '''
    A piece of the full report that we would like to build.
    See :py:func:`full_report`

    :type fold: int

    :param configuration: arbitrary strings to identify individual
                          configuration; we'd typically expect the
                          a configuration to appear over multiple
                          folds
    :type configuration: (string, string...)

    :param predictions: list of edges as you'd get in attelo decode

    :param enable_details: True if we want to enable potentially slower
                         more expensive detailed reporting

    :type enable_details: bool
    '''
    pass


# pylint: disable=too-many-locals
# it's a bit hard to write this sort score accumulation code
# local help
def full_report(mpack, fold_dict, slices):
    """
    Generate a report across a set of folds and configurations.

    This is a bit tricky as the the idea is that we have to acculumate
    per-configuration results over the folds.

    Here configurations are just arbitrary strings

    :param slices: the predictions for each configuration, for each fold.
                   Folds should be contiguous for maximum efficiency.
                   It may be worthwhile to generate this lazily
    :type: slices: iterable(:py:class:`Slice`)
    """
    if not mpack:
        raise ValueError("Can't report with empty multipack")
    edge_count = defaultdict(list)
    edge_lab_count = defaultdict(lambda: defaultdict(list))
    edu_reports = defaultdict(EduReport)
    dpack0 = mpack.values()[0]
    confusion = defaultdict(lambda: empty_confusion_matrix(dpack0))

    fold = None

    for slc in slices:
        if slc.fold != fold:
            f_mpack = select_testing(mpack, fold_dict, slc.fold)
            fpack = DataPack.vstack(f_mpack.values())
            fold = slc.fold
        key = slc.configuration
        # accumulate scores
        edge_count[key].append(score_edges(fpack, slc.predictions))
        edu_reports[key].add(score_edus(fpack, slc.predictions))
        confusion[key] += build_confusion_matrix(fpack, slc.predictions)
        if slc.enable_details:
            details = score_edges_by_label(fpack, slc.predictions)
            for label, lab_count in details:
                edge_lab_count[key][label].append(lab_count)

    edge_report = CombinedReport(EdgeReport,
                                 {k: EdgeReport(v)
                                  for k, v in edge_count.items()})
    # combine
    edge_by_label_report = {}
    for key, counts in edge_lab_count.items():
        report = CombinedReport(LabelReport,
                                {(label,): LabelReport(vs)
                                 for label, vs in counts.items()})
        edge_by_label_report[key] = report

    return ReportPack(edge=edge_report,
                      edge_by_label=edge_by_label_report or None,
                      edu=CombinedReport(EduReport, edu_reports),
                      confusion=confusion,
                      confusion_labels=dpack0.labels)
# pylint: enable=too-many-locals
