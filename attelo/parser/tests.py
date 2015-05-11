"""
attelo.parser tests
"""

from __future__ import print_function

from sklearn.linear_model import (LogisticRegression)
import itertools as itr
import numpy as np
import scipy
import unittest

from attelo.decoding.astar import (AstarArgs,
                                   Heuristic,
                                   RfcConstraint,
                                   AstarDecoder)
from attelo.decoding.baseline import (LastBaseline,
                                      LocalBaseline)
from attelo.decoding.mst import (MstDecoder,
                                 MstRootStrategy)
from attelo.decoding.greedy import (LocallyGreedy)
from attelo.decoding.tests import (DecoderTest)
from attelo.decoding.window import (WindowPruner)

from attelo.edu import EDU, FAKE_ROOT
from attelo.learning.local import (SklearnAttachClassifier,
                                   SklearnLabelClassifier)
from attelo.learning.perceptron import (PerceptronArgs,
                                        StructuredPerceptron)
from attelo.table import (DataPack)
from attelo.util import (Team)

from .full import (JointPipeline,
                   PostlabelPipeline)
from .pipeline import (Pipeline)
from .intra import (for_intra)


# pylint: disable=too-few-public-methods


LOCAL_PERC_ARGS = PerceptronArgs(iterations=20,
                                 averaging=True,
                                 use_prob=False,
                                 aggressiveness=np.inf)
DEFAULT_ASTAR_ARGS = AstarArgs(heuristics=Heuristic.average,
                               rfc=RfcConstraint.none,
                               beam=None,
                               use_prob=True)

# select a decoder and a learner team
MST_DECODER = MstDecoder(root_strategy=MstRootStrategy.fake_root)
ASTAR_DECODER = AstarDecoder(DEFAULT_ASTAR_ARGS)
DECODERS =\
    [
        LastBaseline(),
        LocalBaseline(0.5, use_prob=False),
        MST_DECODER,
        ASTAR_DECODER,
        LocallyGreedy(),
        Pipeline(steps=[('window pruner', WindowPruner(2)),
                        ('decoder', ASTAR_DECODER)]),
    ]

LEARNERS =\
    [
        Team(attach=SklearnAttachClassifier(LogisticRegression()),
             label=SklearnLabelClassifier(LogisticRegression())),

    ]


class ParserTest(DecoderTest):
    """
    Tests for the attelo.parser infrastructure
    """
    def _test_parser(self, parser):
        """
        Train a parser and decode on the same data (not a really
        meaningful test but just trying to make sure we exercise
        as much code as possible)
        """
        target = np.array([1, 2, 3, 1, 4, 3])
        parser.fit([self.dpack], [target])
        parser.transform(self.dpack)

    def test_decoder_by_itself(self):
        for parser in DECODERS:
            self._test_parser(parser)

    def test_joint_parser(self):
        for l, d in itr.product(LEARNERS, DECODERS):
            parser = JointPipeline(learner_attach=l.attach,
                                   learner_label=l.label,
                                   decoder=d)
            self._test_parser(parser)

    def test_postlabel_parser(self):
        learners = LEARNERS +\
            [
                 Team(attach=StructuredPerceptron(MST_DECODER,
                                                  LOCAL_PERC_ARGS),
                      label=SklearnLabelClassifier(LogisticRegression())),
            ]
        for l, d in itr.product(learners, DECODERS):
            parser = PostlabelPipeline(learner_attach=l.attach,
                                       learner_label=l.label,
                                       decoder=d)
            self._test_parser(parser)


class IntraTest(unittest.TestCase):
    """Intrasentential parser"""
    def test_for_intra(self):
        'test that sentence roots are identified correctly'
        # pylint: disable=invalid-name
        a1 = EDU('a1', 'a', 0, 1, 'a', 's1')
        a2 = EDU('a2', 'b', 2, 3, 'a', 's1')
        a3 = EDU('a3', 'c', 4, 5, 'a', 's1')
        b1 = EDU('b1', 'a', 0, 1, 'a', 's2')
        b2 = EDU('b2', 'b', 2, 3, 'a', 's2')
        b3 = EDU('b3', 'c', 4, 5, 'a', 's2')
        # pylint: enable=invalid-name

        orig_classes = ['__UNK__', 'UNRELATED', 'ROOT', 'x']
        dpack = DataPack.load(edus=[a1, a2, a3,
                                    b1, b2, b3],
                              pairings=[(FAKE_ROOT, a1),
                                        (FAKE_ROOT, a2),
                                        (FAKE_ROOT, a3),
                                        (a1, a2),
                                        (a1, a3),
                                        (a2, a3),
                                        (a2, a1),
                                        (a3, a1),
                                        (a3, a2),
                                        (FAKE_ROOT, b1),
                                        (FAKE_ROOT, b2),
                                        (FAKE_ROOT, b3),
                                        (b1, b2),
                                        (b1, b3),
                                        (b2, b3),
                                        (b2, b1),
                                        (b3, b1),
                                        (b3, b2)],
                              data=scipy.sparse.csr_matrix([[1], [1], [1],
                                                            [1], [1], [1],
                                                            [1], [1], [1],
                                                            [1], [1], [1],
                                                            [1], [1], [1],
                                                            [1], [1], [1]]),
                              target=np.array([3, 1, 1, 4, 1, 4, 4, 4, 4,
                                               1, 1, 1, 4, 1, 4, 4, 4, 4]),
                              labels=orig_classes,
                              vocab=None)
        ipack, _ = for_intra(dpack, dpack.target)
        sroots = np.where(ipack.target == ipack.label_number('ROOT'))[0]
        sroot_pairs = ipack.selected(sroots).pairings
        self.assertTrue(all(edu1 == FAKE_ROOT for edu1, edu2 in sroot_pairs),
                        'all root links are roots')
        self.assertEqual(set(e2.subgrouping for _, e2 in sroot_pairs),
                         set(e.subgrouping for e in dpack.edus),
                         'every sentence represented')
