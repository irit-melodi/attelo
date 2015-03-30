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
from attelo.fold import (make_n_fold)
from attelo.io import (load_data_pack,
                       write_predictions_output)
from attelo.report import (CombinedReport,
                           EdgeReport)
from attelo.score import (score_edges)
from attelo.util import (mk_rng, Team)
from sklearn.linear_model import (LogisticRegression)


# pylint: disable=invalid-name

WORKING_DIR = 'example'
PREFIX = fp.join(WORKING_DIR, 'tiny')
TMP_OUTPUT = '/tmp/mini-evaluate'
if not fp.exists(TMP_OUTPUT):
    os.makedirs(TMP_OUTPUT)

# load the data
dpack = load_data_pack(PREFIX + '.edus',
                       PREFIX + '.pairings',
                       PREFIX + '.features.sparse',
                       verbose=True)

# divide the dataset into folds
num_groupings = len(set(e.grouping for e in dpack.edus
                        if e.grouping is not None))
num_folds = min((10, num_groupings))
fold_dict = make_n_fold(dpack, num_folds, mk_rng())

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
    train_pack = dpack.training(fold_dict, fold)
    models = learn(train_pack, learners)

    fold_predictions = []
    # decode each document separately
    for onedoc, indices in dpack.groupings().items():
        print("decoding on file : ", onedoc, file=sys.stderr)
        test_pack = dpack.selected(indices)
        predictions = decode(test_pack, models, decoder,
                             DecodingMode.joint)
        # record the best prediction score (among the decoder nbest)
        doc_scores = [score_edges(test_pack, x) for x in predictions]
        scores.append(max(doc_scores, key=lambda x: x.tpos_label))
        # optional: save the predictions for further inspection
        fold_predictions.extend(predictions[0])

    # optional: write predictions for this fold
    output_file = fp.join(TMP_OUTPUT, 'fold-%d' % fold)
    print("writing: %s" % output_file, file=sys.stderr)
    write_predictions_output(dpack, fold_predictions, output_file)

report = EdgeReport(scores)

# a combined report provides scores for multiple configurations
# here, we are only using it for the single config
combined_report = CombinedReport(EdgeReport,
                                 {('maxent', 'mst'): report})
print(combined_report.table())
