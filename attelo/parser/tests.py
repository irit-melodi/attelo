"""
attelo.parser tests
"""

from __future__ import print_function

from sklearn.linear_model import (LogisticRegression)
import itertools as itr
import numpy as np

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

from attelo.learning.local import (SklearnAttachClassifier,
                                   SklearnLabelClassifier)
from attelo.learning.perceptron import (PerceptronArgs,
                                        StructuredPerceptron)
from attelo.util import (Team)

from .full import (JointPipeline,
                   PostlabelPipeline)
from .pipeline import (Pipeline)


# pylint: disable=too-few-public-methods


LOCAL_PERC_ARGS = PerceptronArgs(iterations=3,
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

