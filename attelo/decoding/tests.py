"""
attelo.decoding tests
"""

from __future__ import print_function
import unittest

from ..edu import EDU
from . import astar

def mk_fake_edu(edu_id, start=0, end=0, edu_file="x"):
    """
    Return a blank EDU object going nowhere
    """
    return EDU(edu_id, start, end, edu_file)


class DecoderTest(unittest.TestCase):
    """
    We could split this into AstarTest, etc
    """
    edus = [mk_fake_edu(x)
            for x in ["x0", "x1", "x2", "x3", "x4"]]

    # would result of prob models max_relation
    # (p(attachement)*p(relation|attachmt))
    prob_distrib =\
            [(edus[1], edus[2], 0.6, 'elaboration'),
             (edus[2], edus[3], 0.3, 'narration'),
             (edus[1], edus[3], 0.4, 'continuation')]
    for one in edus[1:-1]:
        prob_distrib.append((one, edus[4], 0.1, 'continuation'))

    def _test_heuristic(self, heuristic):
        prob = {(a1, a2): (l, p) for a1, a2, p, l in self.prob_distrib}
        pre_heurist = astar.preprocess_heuristics(self.prob_distrib)
        search = astar.DiscourseSearch(heuristic=heuristic.function,
                                       shared={"probs":prob,
                                               "heuristics":pre_heurist,
                                               "use_prob":True,
                                               "RFC": astar.RfcConstraint.full})
        genall = search.launch(astar.DiscData(accessible=[self.edus[1]],
                                              tolink=self.edus[2:]),
                               norepeat=True,
                               verbose=True)
        endstate = genall.next()
        sol = search.recover_solution(endstate)
        #print "solution:", sol
        #print "cost:", endstate.cost()
        #print search.iterations

    def _test_nbest(self, nbest):
        soln = astar.astar_decoder(self.prob_distrib,
                                   astar.AstarArgs(nbest=nbest))
        self.assertEqual(nbest, len(soln))
        return soln

    # TODO: silently crashes, need feedback from Philippe
    #def test_h_average(self):
    #    self._test_heuristic(astar.H_AVERAGE)

    # TODO: currently fails because code returns solution
    # on nbest == 1, or list of solutions otherwise;
    # need feedback from Philippe
    #def test_nbest_1(self):
    #    self._test_nbest(1)

    def test_nbest_2(self):
        self._test_nbest(2)
