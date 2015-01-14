'''
Implementation of the locally greedy approach similar with DuVerle & Predinger
(2009, 2010) (but adapted for SDRT, where the notion of adjacency includes
embedded segments)

July 2012

@author: stergos
'''

from __future__ import print_function
import sys

from .interface import Decoder
from .util import get_sorted_edus, get_prob_map

# pylint: disable=too-few-public-methods


def are_strictly_adjacent(one, two, edus):
    """ returns True in the following cases ::

          [one] [two]
          [two] [one]

        in the rest of the cases (when there is an edu between one and two) it
        returns False
    """
    for edu in edus:
        if edu.id != one.id and edu.id != two.id:
            if one.end <= edu.start and edu.start <= two.start:
                return False
            if one.end <= edu.end and edu.end <= two.start:
                return False
            if two.end <= edu.start and edu.start <= one.start:
                return False
            if two.end <= edu.end and edu.end <= one.start:
                return False

    return True


def is_embedded(one, two):
    """ returns True when one is embedded in two, that is ::

            [two ... [one] ... ]

        returns False in all other cases
    """

    return two.id != one.id and two.start <= one.start and one.end <= two.end


def get_neighbours(edus):
    '''
    Return a mapping from each EDU to its neighbours

    :type edus: [Edu]
    :rtype: Dict Edu [Edu]
    '''

    neighbours = dict()
    for one in edus:
        one_neighbours = []
        one_neighbours_ids = set()
        for two in edus:
            if one.id != two.id:
                if are_strictly_adjacent(one, two, edus):
                    if two.id not in one_neighbours_ids:
                        one_neighbours_ids.add(two.id)
                        one_neighbours.append(two)
                if is_embedded(one, two) or is_embedded(two, one):
                    if two.id not in one_neighbours_ids:
                        one_neighbours_ids.add(two.id)
                        one_neighbours.append(two)

        neighbours[one] = one_neighbours

    return neighbours


class LocallyGreedyState(object):
    '''
    the mutable parts of the locally greedy algorithm
    '''
    def __init__(self, instances):
        self._edus = get_sorted_edus(instances)
        self._edu_ids = set(x.id for x in self._edus)
        self._neighbours = get_neighbours(self._edus)
        self._prob_dist = get_prob_map(instances)

    def _remove_edu(self, original, target):
        '''
        Given a locally greedy state, an original EDU, and a target EDU
        (that the original in meant to point to): remove the original
        edu and merge its neighbourhood into that of the target
        '''
        self._edus.remove(original)
        self._edu_ids.remove(original.id)
        # PM : added to propagate locality to percolated span heads
        tgt_neighbours = self._neighbours[target]
        tgt_neighbours.extend(self._neighbours[original])
        # print(neighbours[new_span], file=sys.stderr)
        tgt_neighbours = [x for x in tgt_neighbours
                          if x.id in self._edu_ids and x.id != target.id]

    def _attach_best(self):
        '''
        Single pass of the locally greedy algorithm: pick the
        highest probability link between any two neighbours.
        Remove the source EDU from future consideration.

        :rtype: None
        '''
        highest = 0.0
        to_remove = None
        attachment = None
        new_span = None

        for source in self._edus:
            for target in self._neighbours[source]:
                if (source.id, target.id) in self._prob_dist:
                    label, prob = self._prob_dist[(source.id, target.id)]
                    if prob > highest:
                        highest = prob
                        to_remove = source
                        new_span = target
                        attachment = (source.id, target.id, label)

        if to_remove is not None:
            self._remove_edu(to_remove, new_span)
            return attachment
        else:  # stop if nothing to attach, but this is wrong
            # print("warning: no attachment found", file=sys.stderr)
            # print(edus)
            # print(edus_id)
            # print([neighbours[x] for x in edus])
            # sys.exit(0)
            self._edus = []
            return None

    def decode(self):
        '''
        Run the decoder

        :rtype [(EDU, EDU, string)]
        '''
        attachments = []
        while len(self._edus) > 1:
            print(len(self._edus), file=sys.stderr)
            attach = self._attach_best()
            if attach is not None:
                attachments.append(attach)
        print("", file=sys.stderr)
        return attachments


# pylint: disable=unused-argument
class LocallyGreedy(Decoder):
    '''
    The locally greedy decoder
    '''
    def __call__(self, instances):
        return [LocallyGreedyState(instances).decode()]
# pylint: enable=unused-argument
