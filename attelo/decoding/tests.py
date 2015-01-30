"""
attelo.decoding tests
"""

from __future__ import print_function
import unittest

from ..edu import EDU
from . import astar, greedy, mst

# pylint: disable=too-few-public-methods


def mk_fake_edu(start, end=None, edu_file="x"):
    """
    Return a blank EDU object going nowhere
    """
    if end is None:
        end = start
    edu_id = 'x{}'.format(start)
    return EDU(edu_id, start, end, edu_file)


class DecoderTest(unittest.TestCase):
    """
    We could split this into AstarTest, etc
    """
    edus = [mk_fake_edu(x)
            for x in range(0, 4)]

    # would result of prob models max_relation
    # (p(attachement)*p(relation|attachmt))
    prob_distrib =\
        [(edus[0], edus[1], 0.6, 'elaboration'),
         (edus[1], edus[2], 0.3, 'narration'),
         (edus[0], edus[2], 0.4, 'continuation')]
    for one in edus[:3]:
        prob_distrib.append((one, edus[3], 0.1, 'acknowledgement'))


class AstarTest(DecoderTest):
    '''tests for the A* decoder'''
    def _test_heuristic(self, heuristic):
        '''
        Run an A* search with the given heuristic
        '''
        prob = {(a1, a2): (l, p) for a1, a2, p, l in self.prob_distrib}
        pre_heurist = astar.preprocess_heuristics(self.prob_distrib)
        config = {"probs": prob,
                  "heuristics": pre_heurist,
                  "use_prob": True,
                  "RFC": astar.RfcConstraint.full}
        search = astar.DiscourseSearch(heuristic=heuristic,
                                       shared=config)
        genall = search.launch(astar.DiscData(accessible=[self.edus[1]],
                                              tolink=self.edus[2:]),
                               norepeat=True,
                               verbose=True)
        endstate = genall.next()
        return search.recover_solution(endstate)
        # print "solution:", sol
        # print "cost:", endstate.cost()
        # print search.iterations

    def _test_nbest(self, nbest):
        'n-best A* search'
        decoder = astar.AstarDecoder(astar.AstarArgs(nbest=nbest))
        soln = decoder.decode(self.prob_distrib)
        self.assertEqual(nbest, len(soln))
        return soln

    # FAILS: it's something to do with the initial state not having
    # any to do links..., would need to check with PM about this
    # def test_h_average(self):
    #     self._test_heuristic(astar.DiscourseState.h_average)

    def test_nbest_1(self):
        '1-best search'
        self._test_nbest(1)

    def test_nbest_2(self):
        '2-best search'
        self._test_nbest(2)


class LocallyGreedyTest(DecoderTest):
    """ Tests for locally greedy decoder"""
    
    def test_locally_greedy(self):
        'check that the locally greedy decoder works'
        decoder = greedy.LocallyGreedy()
        predictions = decoder.decode(self.prob_distrib)
        # made one prediction
        self.assertEqual(1, len(predictions))
        # predicted some attachments in that prediction
        self.assertTrue(predictions[0])

class MstTest(DecoderTest):
    """ Tests for MST and MSDAG decoders """
    
    def test_mst(self):
        edges = mst.MstDecoder().decode(self.prob_distrib)[0]
        # Is it a tree ? (One edge less than number of vertices)
        self.assertEqual(len(edges), len(self.edus) -1)

    def test_msdag(self):
        edges = mst.MsdagDecoder().decode(self.prob_distrib)[0]
        # Are all links included ? (already given a MSDAG...)
        self.assertEqual(len(edges), len(self.prob_distrib))
