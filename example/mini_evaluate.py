"""
Example minature attelo evaluation for a dataset
"""

from __future__ import print_function

from os import path as fp
import os
import sys

from attelo.decoding import (DecodingMode, decode)
from attelo.decoding.mst import (MstDecoder,
                                 MstRootStrategy)
from attelo.learning import (learn)
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
from sklearn.linear_model import (LogisticRegression)


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
                       verbose=True)

# divide the dataset into folds
num_folds = min((10, len(mpack)))
fold_dict = make_n_fold(mpack, num_folds, mk_rng())

# select a decoder and a learner team
decoder = MstDecoder(root_strategy=MstRootStrategy.fake_root)
learners = Team(attach=LogisticRegression(),
                relate=LogisticRegression())

# cross-fold evaluation
scores = []
for fold in range(num_folds):
    print(">>> doing fold ", fold + 1, file=sys.stderr)
    print("training ... ", file=sys.stderr)
    # learn a model for the training data for this fold
    train_pack = select_training(mpack, fold_dict, fold)
    models = learn(train_pack, learners)

    fold_predictions = []
    # decode each document separately
    test_pack = select_testing(mpack, fold_dict, fold)
    for onedoc, dpack in test_pack.items():
        print("decoding on file : ", onedoc, file=sys.stderr)
        predictions = decode(dpack, models, decoder,
                             DecodingMode.joint)
        # record the best prediction score (among the decoder nbest)
        doc_scores = [score_edges(dpack, x) for x in predictions]
        scores.append(max(doc_scores, key=lambda x: x.tpos_label))
        # optional: save the predictions for further inspection
        fold_predictions.extend(predictions[0])

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
