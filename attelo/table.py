"""
Manipulating data tables (taking slices, etc)
"""

from __future__ import print_function
from collections import defaultdict, namedtuple
import numpy
import numpy.ma

from .edu import FAKE_ROOT
from .fold import fold_groupings
from .util import truncate

# pylint: disable=too-few-public-methods

# pylint: disable=pointless-string-statement
UNRELATED = "UNRELATED"
"distinguished value for unrelateted relation labels"
# pylint: enable=pointless-string-statement


class DataPackException(Exception):
    "An exception which arises when worknig with an attelo data pack"

    def __init__(self, msg):
        super(DataPackException, self).__init__(msg)


class DataPack(namedtuple('DataPack',
                          'edus pairings data target labels')):
    '''
    EDUs and features associated with pairs thereof

    :param edus: effectively a set of edus
    :type edus: [:py:class:`EDU`]

    :param pairings: list of edu id pairs
    :type pairings: [(EDU, EDU)]

    :param data: sparse matrix of features, each
                     row corresponding to a pairing

    :param target: array of predictions for each pairing

    :param labels: (optional) list of relation labels
                   (length should be the same as largest value
                   for target)
    :type labels [string] or None
    '''
    # pylint: disable=too-many-arguments
    def __init__(self, edus, pairings, data, target, labels):
        super(DataPack, self).__init__(edus, pairings, data, target, labels)

    @classmethod
    def load(cls, edus, pairings, data, target, labels):
        '''
        Build a data pack and run some sanity checks
        (see :py:method:sanity_check')
               (recommended if reading from disk)

        :rtype: :py:class:`DataPack`
        '''
        pack = cls(edus, pairings, data, target, labels)
        pack.sanity_check()
        return pack
    # pylint: enable=too-many-arguments

    def _check_target(self):
        '''
        sanity check target properties
        '''
        if self.labels is None:
            raise DataPackException('You did not supply any labels in the '
                                    'features file')

        if UNRELATED not in self.labels:
            raise DataPackException('The label "UNRELATED" is missing from '
                                    'the labels list ' + str(self.labels))

        oops = ('The number of labels given ({labels}) is less than '
                'the number of possible target labels ({target}) in '
                'the features file')
        num_classes = len(self.labels)
        max_target = int(max(self.target))
        if num_classes < max_target:
            raise(DataPackException(oops.format(labels=num_classes,
                                                target=max_target)))

    def _check_table_shape(self):
        '''
        sanity check row counts (raises DataPackException)
        '''
        num_insts = self.data.shape[0]
        num_pairings = len(self.pairings)
        num_targets = len(self.target)

        if num_insts != num_pairings:
            oops = ('The number of EDU pairs ({pairs}) does not match '
                    'the number of feature instances ({insts})')
            raise(DataPackException(oops.format(pairs=num_pairings,
                                                insts=num_insts)))

        if num_insts != num_targets:
            oops = ('The number of target elements ({tgts}) does not match '
                    'the number of feature instances ({insts})')
            raise(DataPackException(oops.format(tgts=num_targets,
                                                insts=num_insts)))

    def groupings(self):
        '''
        return a dictionary of table indices corresponding to the items
        for all groupings; likely but not necessarily consequenctive

        (raises DataPackException if we have pairings that traverse
        group boundaries, which is a no-no in the attelo model)

        :rtype: dict(string, [int])
        '''
        res = defaultdict(list)
        for i, (edu1, edu2) in enumerate(self.pairings):
            grp1 = edu1.grouping
            grp2 = edu2.grouping
            if grp1 is None:
                grp = grp2
            elif grp2 is None:
                grp = grp1
            elif grp1 != grp2:
                oops = ('Grouping mismatch: {edu1} is in group {grp1}, '
                        'but {edu2} is in {grp2}')
                raise(DataPackException(oops.format(edu1=edu1,
                                                    edu2=edu2,
                                                    grp1=grp1,
                                                    grp2=grp2)))
            else:
                grp = grp1
            res[grp].append(i)
        return res

    def sanity_check(self):
        '''
        Raising :py:class:`DataPackException` if anything about
        this datapack seems wrong, for example if the number of
        rows in one table is not the same as in another
        '''
        self._check_target()
        self._check_table_shape()
        self.groupings()

    def selected(self, indices):
        '''
        Return only the items in the specified rows
        '''
        # pylint: disable=no-member
        sel_targets = numpy.take(self.target, indices)
        # pylint: enable=no-member
        if self.labels is None:
            sel_labels = None
        else:
            sel_labels = self.labels
        sel_pairings = [self.pairings[x] for x in indices]
        sel_edus_ = set()
        for edu1, edu2 in sel_pairings:
            sel_edus_.add(edu1)
            sel_edus_.add(edu2)
        sel_edus = [e for e in self.edus if e in sel_edus_]
        sel_data = self.data[indices]
        return DataPack(edus=sel_edus,
                        pairings=sel_pairings,
                        data=sel_data,
                        target=sel_targets,
                        labels=sel_labels)

    def _select_fold(self, fold_dict, pred):
        '''
        Given a division into folds and a predicate on folds,
        return only the items for which the fold predicate is
        True

        :rtype: :py:class:`DataPack`
        '''
        group_indices = self.groupings()
        indices = []
        for key, val in fold_dict.items():
            if pred(val):
                idx = group_indices[key]
                indices.extend(idx)
        return self.selected(indices)

    def training(self, fold_dict, fold):
        '''
        Given a division into folds and a fold number,
        return only the training items for that fold

        :rtype: :py:class:`DataPack`
        '''
        fold_groupings(fold_dict, fold)  # sanity check
        return self._select_fold(fold_dict, lambda x: x != fold)

    def testing(self, fold_dict, fold):
        '''
        Given a division into folds and a fold number,
        return only the test items for that fold

        :rtype: :py:class:`DataPack`
        '''
        fold_groupings(fold_dict, fold)  # sanity check
        return self._select_fold(fold_dict, lambda x: x == fold)

    def attached_only(self):
        '''
        Return only the instances which are labelled as
        attached (ie. this would presumably return an empty
        pack on completely unseen data)
        '''
        # pylint: disable=no-member
        unrelated = self.label_number(UNRELATED)
        indices = numpy.where(self.target != unrelated)[0]
        # pylint: enable=no-member
        return self.selected(indices)

    def get_label(self, i):
        '''
        Return the class label for the given target value.

        If `labels` is None, return `None`
        '''
        if self.labels is None:
            return None
        else:
            return self.labels[int(i) - 1]

    def label_number(self, label):
        '''
        Return the numerical label that corresponnds to the given
        string label

        :rtype: float
        '''
        return self.labels.index(label) + 1


def for_attachment(pack):
    '''
    Adapt a datapack to the attachment task. This could involve

        * selecting some of the features (all for now, but may
          change in the future)
        * modifying the features/labels in some way
          (we binarise them to 0 vs not-0)

    :rtype: :py:class:`DataPack`
    '''
    # pylint: disable=no-member
    unrelated = pack.label_number(UNRELATED)
    tweak = numpy.vectorize(lambda x: -1 if x == unrelated else 1)
    # pylint: enable=no-member
    return DataPack(edus=pack.edus,
                    pairings=pack.pairings,
                    data=pack.data,
                    target=tweak(pack.target),
                    labels=None)


def for_labelling(pack):
    '''
    Adapt a datapack to the relation labelling task (currently a no-op).
    This could involve

        * selecting some of the features (all for now, but may
          change in the future)
        * modifying the features/labels in some way (in practice
          no change)

    :rtype: :py:class:`DataPack`
    '''
    return pack
