"""
Manipulating data tables (taking slices, etc)
"""

from __future__ import print_function
from collections import defaultdict, namedtuple

import numpy
import numpy.ma
import scipy.sparse

from .edu import FAKE_ROOT_ID
from .util import concat_l

# pylint: disable=too-few-public-methods

# pylint: disable=pointless-string-statement
UNRELATED = "UNRELATED"
"distinguished value for unrelateted relation labels"

UNLABELLED = "unlabelled"
"distinguished internal value for post-labelling mode"
# pylint: enable=pointless-string-statement


class DataPackException(Exception):
    "An exception which arises when worknig with an attelo data pack"

    def __init__(self, msg):
        super(DataPackException, self).__init__(msg)


class DataPack(namedtuple('DataPack',
                          'edus pairings data target labels')):
    '''
    A set of data that can be said to belong together.

    A typical use of the datapack would be to group together
    data for a single document/grouping. But in cases where
    this distinction does not matter, it can also be convenient
    to combine data from multiple documents into a single pack.

    :param edus: effectively a set of edus
    :type edus: [:py:class:`EDU`]

    :param pairings: list of edu id pairs
    :type pairings: [(EDU, EDU)]

    :param data: sparse matrix of features, each
                 row corresponding to a pairing

    :param target: array of predictions for each pairing

    :param labels: list of relation labels
                   (length should be the same as largest value
                   for target)
    :type labels: [string]
    '''
    # pylint: disable=too-many-arguments
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

    @classmethod
    def vstack(cls, dpacks):
        '''
        Combine several datapacks into one.

        The labels for all packs must be the same

        :type dpacks: [DataPack]
        '''
        if not dpacks:
            raise ValueError('need non-empty list of datapacks')
        # pylint: disable=no-member
        return DataPack(edus=concat_l(d.edus for d in dpacks),
                        pairings=concat_l(d.pairings for d in dpacks),
                        data=scipy.sparse.vstack(d.data for d in dpacks),
                        target=numpy.concatenate([d.target for d in dpacks]),
                        labels=dpacks[0].labels)
        # pylint: enable=no-member

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

    def sanity_check(self):
        '''
        Raising :py:class:`DataPackException` if anything about
        this datapack seems wrong, for example if the number of
        rows in one table is not the same as in another
        '''
        self._check_target()
        self._check_table_shape()

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
        '''
        return get_label_string(self.labels, i)

    def label_number(self, label):
        '''
        Return the numerical label that corresponnds to the given
        string label

        :rtype: float
        '''
        return self.labels.index(label) + 1


def groupings(pairings):
    '''
    Given a list of EDU pairings, return a dictionary mapping
    grouping names to list of rows within the pairings.

    :rtype: dict(string, [int])
    '''
    res = defaultdict(list)
    for i, (edu1, edu2) in enumerate(pairings):
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


def select_intrasentential(dpack, include_fake_root=True):
    """
    Retain only the pairings from a datapack which correspond to
    EDUs in the same sentence (or the fake root).

    Note that both `select_intrasentential` and
    `select_intersentential` include the fake root EDUs
    """
    retain = []
    for i, (edu1, edu2) in enumerate(dpack.pairings):
        if edu1.id == FAKE_ROOT_ID:
            if include_fake_root:
                retain.append(i)
        elif edu1.subgrouping == edu2.subgrouping:
            retain.append(i)
    return dpack.selected(retain)


def select_intersentential(dpack, include_fake_root=True):
    """
    Retain only the pairings from a datapack which correspond to
    EDUs in different sentences (or the fake root).

    Note that both `select_intrasentential` and
    `select_intersentential` include the fake root EDUs
    """
    retain = []
    for i, (edu1, edu2) in enumerate(dpack.pairings):
        if edu1.id == FAKE_ROOT_ID:
            if include_fake_root:
                retain.append(i)
        elif edu1.subgrouping != edu2.subgrouping:
            retain.append(i)
    return dpack.selected(retain)


def for_intra(pack):
    '''
    Adapt a datapack to intrasentential decoding. An intrasenential
    datapack is almost identical to its original, except that we
    set the label for each ('ROOT', edu) pairing to 'ROOT' if that
    edu is a subgrouping head (if it has no parents other than than
    'ROOT' within its subgrouping).

    This should be done before either :py:func:`for_labelling` or
    :py:func:`for_attachment`

    :rtype: :py:class:`DataPack`
    '''
    # pack = _select_intrasentential(pack)
    local_heads = defaultdict(set)
    ruled_out = defaultdict(set)
    indices = {}
    for i, (edu1, edu2) in enumerate(pack.pairings):
        subgrouping = edu1.subgrouping or edu2.subgrouping
        if edu1.id == FAKE_ROOT_ID:
            if edu2.id not in ruled_out[subgrouping]:
                local_heads[subgrouping].add(edu2.id)
                indices[edu2.id] = i
        else:
            # any child edu is necessarily ruled out
            ruled_out[subgrouping].add(edu2.id)
            if edu2.id in local_heads[subgrouping]:
                local_heads[subgrouping].remove(edu2.id)

    all_heads = []
    for subgrouping, heads in local_heads.items():
        all_heads.extend(indices[x] for x in heads
                         if x not in ruled_out[subgrouping])

    # pylint: disable=no-member
    new_target = numpy.copy(pack.target)
    new_target[all_heads] = pack.label_number('ROOT')
    # pylint: enable=no-member
    return DataPack(edus=pack.edus,
                    pairings=pack.pairings,
                    data=pack.data,
                    target=new_target,
                    labels=pack.labels)


class Multipack(dict):
    '''
    A multipack is a mapping from groupings to datapacks

    This class exists purely for documentation purposes; in
    practice, a dictionary of string to Datapack will do just
    fine
    '''
    pass


def select_window(dpack, window):
    '''Select only EDU pairs that are at most `window` EDUs apart
    from each other (adjacent EDUs would be considered `0` apart)

    Note that if the window is `None`, we simply return the
    original datapack

    Note that will only work correctly on single-document datapacks
    '''
    if window is None:
        return dpack
    position = {FAKE_ROOT_ID: 0}
    for i, edu in enumerate(dpack.edus):
        position[edu.id] = i
    indices = []
    for i, (edu1, edu2) in enumerate(dpack.pairings):
        gap = abs(position[edu2.id] - position[edu1.id])
        if gap <= window:
            indices.append(i)
    return dpack.selected(indices)


def get_label_string(labels, i):
    '''
    Return the class label for the given target value.
    '''
    return labels[int(i) - 1]
