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
                     score_edus,
                     select_in_pack)
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

    def _filenames(self, output_dir):
        """
        Return a dictionary of filenames; keys are internal
        to this class
        """
        makedirs(output_dir)
        filenames = {}

        if self.edge is not None:
            # edgewise scores
            filenames['edge'] = fp.join(output_dir, 'scores.txt')

        if self.edu is not None:
            # edu scores
            filenames['edu'] = fp.join(output_dir, 'edu-scores.txt')

        if self.edge_by_label is not None:
            label_dir = fp.join(output_dir, 'label-scores')
            makedirs(label_dir)
            for key in self.edge_by_label:
                fkey = ('label', key)
                filenames[fkey] = fp.join(label_dir, '-'.join(key))

        if self.confusion is not None:
            confusion_dir = fp.join(output_dir, 'confusion')
            makedirs(confusion_dir)
            for key in self.confusion:
                fkey = ('confusion', key)
                filenames[fkey] = fp.join(confusion_dir, '-'.join(key))
        return filenames

    def dump(self, output_dir, header=None):
        """
        Save reports to an output directory
        """
        makedirs(output_dir)
        fnames = self._filenames(output_dir).values()
        # touch every file
        for fname in fnames:
            with open(fname, 'w'):
                pass
        self.append(output_dir, header=header)

    def append(self, output_dir, header):
        """
        Append reports to a pre-existing output directory

        :param header: optional name for the tables
        :type header: string or None
        """
        fnames = self._filenames(output_dir)
        if self.edge is not None:
            # edgewise scores
            with open(fnames['edge'], 'a') as ostream:
                print(self.edge.table(main_header=header), file=ostream)
                print(file=ostream)

        if self.edu is not None:
            # edu scores
            with open(fnames['edu'], 'a') as ostream:
                print(self.edu.table(main_header=header), file=ostream)
                print(file=ostream)

        if self.edge_by_label is not None:
            for key, report in self.edge_by_label.items():
                fkey = ('label', key)
                with open(fnames[fkey], 'a') as ostream:
                    print(report.table(sortkey=lambda (_, v): 0 - v.count,
                                       main_header=header),
                          file=ostream)
                    print(file=ostream)

        if self.confusion is not None:
            for key, matrix in self.confusion.items():
                fkey = ('confusion', key)
                with open(fnames[fkey], 'a') as ostream:
                    print(show_confusion_matrix(self.confusion_labels, matrix,
                                                main_header=header),
                          file=ostream)
                    print(file=ostream)


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
def full_report(mpack, fold_dict, slices,
                adjust_pack=None):
    """
    Generate a report across a set of folds and configurations.

    This is a bit tricky as the the idea is that we have to acculumate
    per-configuration results over the folds.

    Here configurations are just arbitrary strings

    :param slices: the predictions for each configuration, for each fold.
                   Folds should be contiguous for maximum efficiency.
                   It may be worthwhile to generate this lazily
    :type slices: iterable(:py:class:`Slice`)

    :param adjust_pack: (optional) function that modifies a DataPack, for
                        example by picking out a subset of the pairings.
    :type adjust_pack: (DataPack -> DataPack) or None
    """
    if not mpack:
        raise ValueError("Can't report with empty multipack")
    edge_count = defaultdict(list)
    edge_lab_count = defaultdict(lambda: defaultdict(list))
    edu_reports = defaultdict(EduReport)
    dpack0 = mpack.values()[0]
    confusion = defaultdict(lambda: empty_confusion_matrix(dpack0))

    fold = None

    # avoid slicing the predictions if we can help it (slow)
    if adjust_pack is None:
        adjust_predictions = lambda _, x: x
        adjust_pack = lambda x: x
    else:
        adjust_predictions = select_in_pack

    for slc in slices:
        if slc.fold != fold:
            f_mpack = select_testing(mpack, fold_dict, slc.fold)
            fpack = DataPack.vstack([adjust_pack(x)
                                     for x in f_mpack.values()])
            fold = slc.fold
        key = slc.configuration
        # accumulate scores
        predictions = adjust_predictions(fpack, slc.predictions)
        edge_count[key].append(score_edges(fpack, predictions))
        edu_reports[key].add(score_edus(fpack, predictions))
        confusion[key] += build_confusion_matrix(fpack, predictions)
        details = score_edges_by_label(fpack, predictions)
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
