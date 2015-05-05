"""
attelo.decoding tests
"""

from __future__ import print_function
import unittest

import numpy as np
from scipy.sparse import csr_matrix

from ..table import (DataPack, Graph)
from ..edu import EDU
from . import astar, greedy, mst
from .astar import (AstarArgs, Heuristic, RfcConstraint)
from .util import (prediction_to_triples, simple_candidates)

# pylint: disable=too-few-public-methods
# default values for perceptron learner

#DEFAULT_USE_PROB = True
#DEFAULT_PERCEPTRON_ARGS = PerceptronArgs(iterations=20,
#                                         averaging=True,
#                                         use_prob=DEFAULT_USE_PROB,
#                                         aggressiveness=inf)

# DEFAULT_MST_ROOT = MstRootStrategy.fake_root

# default values for A* decoder
# (NB: not the same as in the default initialiser)
DEFAULT_ASTAR_ARGS = AstarArgs(heuristics=Heuristic.average,
                               rfc=RfcConstraint.full,
                               beam=None,
                               use_prob=True)
DEFAULT_HEURISTIC = DEFAULT_ASTAR_ARGS.heuristics
DEFAULT_BEAMSIZE = DEFAULT_ASTAR_ARGS.beam
DEFAULT_RFC = DEFAULT_ASTAR_ARGS.rfc


def mk_fake_edu(start, end=None, edu_file="x", sentence='s1'):
    """
    Return a blank EDU object going nowhere
    """
    if end is None:
        end = start
    edu_id = 'x{}'.format(start)
    return EDU(edu_id, edu_id, start, end, edu_file, sentence)


class DecoderTest(unittest.TestCase):
    """
    We could split this into AstarTest, etc
    """
    edus = [mk_fake_edu(x)
            for x in range(0, 4)]

    # would result of prob models max_relation
    # (p(attachement)*p(relation|attachmt))
    pairings = [(edus[0], edus[1]),
                (edus[1], edus[2]),
                (edus[0], edus[2]),
                (edus[0], edus[3]),
                (edus[1], edus[3]),
                (edus[2], edus[3])]
    graph = Graph(prediction=np.array([0, 0, 0, 0, 0, 0]),
                  attach=np.array([0.8, 0.4, 0.5, 0.2, 0.2, 0.2]),
                  label=np.array([[0.0, 0.1, 0.8, 0.1, 0.0],
                                  [0.0, 0.2, 0.1, 0.9, 0.0],
                                  [0.0, 0.1, 0.0, 0.0, 0.2],
                                  [0.0, 0.0, 0.0, 0.0, 0.7],
                                  [0.0, 0.0, 0.0, 0.0, 0.7],
                                  [0.0, 0.0, 0.0, 0.0, 0.7]]))
    dpack = DataPack(labels=['__UNK__',
                             'UNRELATED',
                             'elaboration',
                             'narration',
                             'acknowledgement'],
                     edus=edus,
                     pairings=pairings,
                     data=csr_matrix(np.array([[1, 3, 8],
                                               [9, 2, 0],
                                               [7, 9, 1],
                                               [3, 1, 1],
                                               [3, 3, 1],
                                               [0, 6, 3]])),
                     target=np.array([0, 0, 0, 0, 0, 0]),
                     graph=graph,
                     vocab=None)


class AstarTest(DecoderTest):
    '''tests for the A* decoder'''
    def _test_heuristic(self, heuristic):
        '''
        Run an A* search with the given heuristic
        '''
        cands = simple_candidates(self.dpack)
        prob = {(a1, a2): (l, p) for a1, a2, p, l in cands}
        pre_heurist = astar.preprocess_heuristics(cands)
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

    def test_search(self):
        'n-best A* search'
        astar_args = astar.AstarArgs(heuristics=DEFAULT_ASTAR_ARGS.heuristics,
                                     # FIXME full broken
                                     rfc=astar.RfcConstraint.simple,
                                     beam=DEFAULT_ASTAR_ARGS.beam,
                                     use_prob=DEFAULT_ASTAR_ARGS.use_prob)
        decoder = astar.AstarDecoder(astar_args)
        return decoder.decode(self.dpack)

    # FAILS: it's something to do with the initial state not having
    # any to do links..., would need to check with PM about this
    # def test_h_average(self):
    #     self._test_heuristic(astar.DiscourseState.h_average)


class LocallyGreedyTest(DecoderTest):
    """ Tests for locally greedy decoder"""

    def test_locally_greedy(self):
        'check that the locally greedy decoder works'
        decoder = greedy.LocallyGreedy()
        decoder.decode(self.dpack)


class MstTest(DecoderTest):
    """ Tests for MST and MSDAG decoders """

    def test_mst(self):
        'check plain MST decoder'
        decoder1 = mst.MstDecoder(mst.MstRootStrategy.fake_root)
        edges = prediction_to_triples(decoder1.decode(self.dpack))
        # Is it a tree ? (One edge less than number of vertices)
        self.assertEqual(len(edges), len(self.edus) - 1)

        decoder2 = mst.MstDecoder(mst.MstRootStrategy.leftmost)
        edges = prediction_to_triples(decoder2.decode(self.dpack))
        # Is it a tree ? (One edge less than number of vertices)
        self.assertEqual(len(edges), len(self.edus) - 1)

    def test_msdag(self):
        'check MSDAG decoder'
        decoder = mst.MsdagDecoder(mst.MstRootStrategy.fake_root)
        edges = prediction_to_triples(decoder.decode(self.dpack))
        # Are all links included ? (already given a MSDAG...)
        self.assertEqual(len(edges), len(self.dpack))
