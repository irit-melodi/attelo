"""Example test harness
"""

from __future__ import print_function
from tempfile import mkdtemp
from os import path as fp
import glob
import shutil

from sklearn.linear_model import (LogisticRegression)

from .config import (EvaluationConfig,
                     Keyed,
                     LearnerConfig,
                     RuntimeConfig)
from .evaluate import (evaluate_corpus,
                       prepare_dirs)
from .interface import Harness
from attelo.decoding.mst import (MstDecoder,
                                 MstRootStrategy)
from attelo.decoding.greedy import (LocallyGreedy)

from attelo.learning.local import (SklearnAttachClassifier,
                                   SklearnLabelClassifier)

from attelo.parser.full import (JointPipeline,
                                PostlabelPipeline)
import attelo.fold


# pylint: disable=too-few-public-methods

class TinyHarness(Harness):
    """Example harness that runs on the example data
    """
    _maxent_a = Keyed('maxent',
                      SklearnAttachClassifier(LogisticRegression()))
    _maxent_l = Keyed('maxent',
                      SklearnLabelClassifier(LogisticRegression()))
    _maxent = LearnerConfig(attach=_maxent_a,
                            label=_maxent_l)
    _decoder1 = MstDecoder(root_strategy=MstRootStrategy.fake_root)
    _decoder2 = LocallyGreedy()
    _parser1 = Keyed("mst-j",
                     JointPipeline(_maxent.attach.payload,
                                   _maxent.label.payload,
                                   _decoder1))
    _parser2 = Keyed("greedy-p",
                     PostlabelPipeline(_maxent.attach.payload,
                                       _maxent.label.payload,
                                       _decoder2))
    _evaluations = [EvaluationConfig(key="maxent-mst-j",
                                     settings=Keyed('j', None),
                                     learner=_maxent,
                                     parser=_parser1),
                    EvaluationConfig(key="maxent-greedy-p",
                                     settings=Keyed('p', None),
                                     learner=_maxent,
                                     parser=_parser2)]

    def __init__(self):
        self._basedir = mkdtemp()
        for cpath in glob.glob('doc/example-corpus/*'):
            shutil.copy(cpath, self._basedir)
        super(TinyHarness, self).__init__('tiny', None)

    def run(self):
        """Run the evaluation
        """
        runcfg = RuntimeConfig.empty()
        eval_dir, scratch_dir = prepare_dirs(runcfg, self._basedir)
        self.load(runcfg, eval_dir, scratch_dir)
        evaluate_corpus(self)

    @property
    def evaluations(self):
        return self._evaluations

    @property
    def test_evaluation(self):
        return None

    def create_folds(self, mpack):
        return attelo.fold.make_n_fold(mpack, 2, None)

    def mpack_paths(self, _, stripped=False):
        """Return a dict of paths needed to read a datapack.

        The 2nd argument denoted by '_' is test_data, which is unused in
        this example.
        """
        core_path = fp.join(self._basedir, 'data', 'tiny')
        return {'edu_input': core_path + '.edus',
                'pairings': core_path + '.pairings',
                'features': core_path + '.features.sparse',
                'vocab': core_path + '.features.sparse.vocab',
                'labels': core_path + '.labels'}

    def _model_basename(self, rconf, mtype, ext):
        "Basic filename for a model"

        if 'attach' in mtype:
            rsubconf = rconf.attach
        else:
            rsubconf = rconf.label

        template = '{dataset}.{learner}.{task}.{ext}'
        return template.format(dataset=self.dataset,
                               learner=rsubconf.key,
                               task=mtype,
                               ext=ext)

    def model_paths(self, rconf, fold, parser):
        if fold is None:
            parent_dir = self.combined_dir_path()
        else:
            parent_dir = self.fold_dir_path(fold)

        def _eval_model_path(mtype):
            "Model for a given loop/eval config and fold"
            bname = self._model_basename(rconf, mtype, 'model')
            return fp.join(parent_dir, bname)

        return {'attach': _eval_model_path("attach"),
                'label': _eval_model_path("relate")}
