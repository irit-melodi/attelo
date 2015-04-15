"""
Example minature attelo evaluation for a dataset
"""

from __future__ import print_function

from os import path as fp
import os
import sys

from sklearn.linear_model import (LogisticRegression)
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
from attelo.decoding.util import (prediction_to_triples)
from attelo.decoding.window import (WindowPruner)

from attelo.learning.local import (SklearnAttachClassifier,
                                   SklearnLabelClassifier)
from attelo.learning.perceptron import (PerceptronArgs,
                                        StructuredPerceptron)
from attelo.parser.full import (JointPipeline,
                                PostlabelPipeline)
from attelo.parser.label import (SimpleLabeller)
from attelo.parser.pipeline import (Pipeline)

from attelo.fold import (make_n_fold,
                         select_testing,
                         select_training)
from attelo.io import (load_multipack,
                       write_predictions_output)
from attelo.report import (CombinedReport,
                           EdgeReport)
from attelo.score import (score_edges)
from attelo.table import (DataPack)
from attelo.util import (mk_rng, Team)

# pylint: disable=invalid-name

WORKING_DIR = 'example'
PREFIX = fp.join(WORKING_DIR, 'tiny')
TMP_OUTPUT = '/tmp/mini-evaluate'
if not fp.exists(TMP_OUTPUT):
    os.makedirs(TMP_OUTPUT)

# load the data
mpack = load_multipack(PREFIX + '.edus',
                       PREFIX + '.pairings',
                       PREFIX + '.features.sparse',
                       PREFIX + '.features.sparse.vocab',
                       verbose=True)

# divide the dataset into folds
num_folds = min((10, len(mpack)))
fold_dict = make_n_fold(mpack, num_folds, mk_rng())

LOCAL_PERC_ARGS = PerceptronArgs(iterations=20,
                                 averaging=True,
                                 use_prob=False,
                                 aggressiveness=np.inf)
DEFAULT_ASTAR_ARGS = AstarArgs(heuristics=Heuristic.average,
                               rfc=RfcConstraint.none,
                               beam=None,
                               use_prob=True)

# select a decoder and a learner team
decoder = Pipeline(steps=[('window pruner', WindowPruner(2)),
                          ('decoder', AstarDecoder(DEFAULT_ASTAR_ARGS))])
# decoder = LocallyGreedy()
# decoder = MstDecoder(root_strategy=MstRootStrategy.fake_root)
# decoder = LocalBaseline(0.5, use_prob=False)
learners = Team(attach=StructuredPerceptron(decoder, LOCAL_PERC_ARGS),
                relate=SklearnLabelClassifier(LogisticRegression()))
#parser = JointPipeline(learner_attach=learners.attach,
#                       learner_label=learners.relate,
#                       decoder=decoder)
parser = PostlabelPipeline(learner_attach=learners.attach,
                           learner_label=learners.relate,
                           decoder=decoder)

# cross-fold evaluation
scores = []
for fold in range(num_folds):
    print(">>> doing fold ", fold + 1, file=sys.stderr)
    print("training ... ", file=sys.stderr)
    # learn a model for the training data for this fold
    train_packs = select_training(mpack, fold_dict, fold).values()
    parser.fit(train_packs,
               [x.target for x in train_packs])

    fold_predictions = []
    # decode each document separately
    test_pack = select_testing(mpack, fold_dict, fold)
    for onedoc, dpack in test_pack.items():
        print("decoding on file : ", onedoc, file=sys.stderr)
        dpack = parser.transform(dpack)
        prediction = prediction_to_triples(dpack)
        print("Predictions: ", prediction)
        # record the prediction score
        scores.append(score_edges(dpack, prediction))
        # optional: save the predictions for further inspection
        fold_predictions.extend(prediction)

    # optional: write predictions for this fold
    output_file = fp.join(TMP_OUTPUT, 'fold-%d' % fold)
    print("writing: %s" % output_file, file=sys.stderr)
    write_predictions_output(DataPack.vstack(test_pack.values()),
                             fold_predictions, output_file)

report = EdgeReport(scores)

# a combined report provides scores for multiple configurations
# here, we are only using it for the single config
combined_report = CombinedReport(EdgeReport,
                                 {('maxent', 'mst'): report})
print(combined_report.table())
