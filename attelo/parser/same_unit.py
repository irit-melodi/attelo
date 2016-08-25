"""Preprocessor to detect and link fragments of EDus ("same-unit").
"""

from __future__ import print_function

from os import path as fp

import csv
import joblib
import numpy as np
import os

# WIP 2016-08-25 nary fragmented EDUs
from educe.rst_dt.deptree import binary_to_nary
# end nary fragmented EDUs

from attelo.edu import edu_id2num
from attelo.table import UNKNOWN, DataPack, Graph
from attelo.learning.interface import AttachClassifier
from attelo.learning.local import SklearnClassifier
from attelo.parser.attach import AttachClassifierWrapper
from attelo.parser.full import AttachTimesBestLabel
from attelo.parser.label import LabelClassifierWrapper
from attelo.parser.interface import Parser
from attelo.parser.pipeline import Pipeline


SAME_UNIT = "same-unit"


def for_attachment_same_unit(dpack, target):
    """Adapt a datapack to the task of linking fragments of EDUs.

    This is modelled as an attachment task restricted to the "same-unit"
    label.

    This could involve:
    * selecting some of the features (all for now, but may change in the
    future)
    * modifying the features/labels in some way: we currently binarise
    labels to {-1 ; 1} for UNRELATED and not-UNRELATED respectively.

    Parameters
    ----------
    dpack: DataPack
        Original datapack
    target: array(int)
        Original targets

    Returns
    -------
    dpack: DataPack
        Transformed datapack, with binary labels

    target: array(int)
        Transformed targets, with binary labels
    """
    su_idx = dpack.label_number(SAME_UNIT)
    dpack = DataPack(edus=dpack.edus,
                     pairings=dpack.pairings,
                     data=dpack.data,
                     target=np.where(dpack.target == su_idx, 1, -1),
                     # WIP
                     ctarget=dpack.ctarget,  # old WIP
                     # 2016-07-28 CDUs
                     cdus=dpack.cdus,
                     cdu_pairings=dpack.cdu_pairings,
                     cdu_data=dpack.cdu_data,
                     cdu_target=dpack.cdu_target,
                     # end WIP
                     labels=[UNKNOWN, SAME_UNIT],
                     vocab=dpack.vocab,
                     graph=dpack.graph)
    target = np.where(target == su_idx, 1, -1)
    return dpack, target


def right_intra_idc(dpack):
    """Get the indices of right-attachment, intra-sentential pairings.

    Parameters
    ----------
    dpack: DataPack
        Datapack

    Returns
    -------
    res: array of integers
        Indices of right, intra-sentential candidates in dpack.pairings.
    """
    edu_id2sent = {e.id: e.subgrouping for e in dpack.edus}
    res = [i for i, (edu1, edu2) in enumerate(dpack.pairings)
           if (edu_id2sent[edu1.id] == edu_id2sent[edu2.id]
               and edu_id2num(edu1.id) < edu_id2num(edu2.id))]
    return res


class SklearnSameUnitClassifier(AttachClassifier, SklearnClassifier):
    """A relatively simple way to get a "same-unit" classifier:
    just pass in an sklearn classifier.

    Parameters
    ----------
    learner: sklearn API-compatible classifier
        The learner to use for label prediction.

    pos_label: str or int, 1 by default
        The class that codes an attachment decision for "same-unit".

    Notes
    -----
    This is almost identical to what should be the clean version of
    SklearnAttachClassifier. When the latter is clean, this class
    should disappear and be replaced in callers by SklearnAttachClassifier.
    """

    def __init__(self, learner, pos_label=1):
        AttachClassifier.__init__(self)
        SklearnClassifier.__init__(self, learner)
        self._fitted = False
        self.pos_label = pos_label

    def fit(self, dpacks, targets):
        dpack = DataPack.vstack(dpacks)
        target = np.concatenate(targets)
        if dpack.pairings:
            # don't call fit if there is no instance ; as of 2016-08-24,
            # this happens in an IntraInterParser, because we restrict
            # candidate same-units to belong to the same sentence
            self._learner.fit(dpack.data, target)
        self._fitted = True
        return self

    def predict_score(self, dpack):
        """Predict "same-unit" attachment score.

        Parameters
        ----------
        dpack: DataPack
            Original DataPack

        Returns
        -------
        scores_pred: array of float
            Attachment scores for "same-unit" for each pairing of dpack.
        """
        if not self._fitted:
            raise ValueError('Fit not yet called')

        if self.can_predict_proba:
            attach_idx = list(self._learner.classes_).index(self.pos_label)
            probs = self._learner.predict_proba(dpack.data)
            scores_pred = probs[:, attach_idx]
        else:
            scores_pred = self._learner.decision_function(dpack.data)

        return scores_pred


class SameUnitClassifierWrapper(Parser):
    """
    Parser that extracts attachments weights from a "same-unit"
    classifier.

    This parser is really meant to be used in conjunction with
    other parsers downstream that make use of these weights.

    If you use it in standalone mode, it will just provide the
    standard unknown prediction everywhere

    Notes
    -----
    *Cache keys*

    * attach: attachment model path
    """
    def __init__(self, learner_su):
        """
        Parameters
        ----------
        learner_su : SklearnSameUnitClassifier
            Learner to use for "same-unit".
        """
        self._learner_su = learner_su

    def fit(self, dpacks, targets, nonfixed_pairs=None, cache=None):
        """
        Extract whatever models or other information from the multipack
        that is necessary to make the parser operational

        Parameters
        ----------
        dpacks: list of DataPack
            List of datapacks, one per "stuctured instance" (ex:
            document or sentence for RST, document, dialogue or
            turn for STAC).

        targets: list of array of int
            List of arrays of gold labels, one array per structured
            instance, then one integer (label) per candidate edge.

        nonfixed_pairs: list of array of int
            List of arrays of indexes, corresponding to the non-fixed
            candidate edges in each instance.

        cache: TODO
            TODO

        Returns
        -------
        self: SameUnitClassifierWrapper
            Fitted self.
        """
        cache_file = (cache.get('su') if cache is not None
                      else None)
        # load cached classifier, if it exists
        if cache_file is not None and fp.exists(cache_file):
            # print('\tload {}'.format(cache_file))
            self._learner_su = joblib.load(cache_file)
            return self

        dpacks, targets = self.dzip(for_attachment_same_unit,
                                    dpacks, targets)

        # WIP filter: pass only nonfixed, right-attachment, intra-sentential
        # pairs to the classifier
        ri_pairs = [right_intra_idc(x) for x in dpacks]
        if nonfixed_pairs is not None:
            nf_pairs = [list(np.intersect1d(rip, nfp)) for rip, nfp
                        in zip(ri_pairs, nonfixed_pairs)]
        else:
            nf_pairs = ri_pairs

        dpacks = [dpack.selected(nfp) for dpack, nfp
                  in zip(dpacks, nf_pairs)]
        targets = [target[nfp] for target, nfp
                   in zip(targets, nf_pairs)]
        # end filter

        self._learner_su.fit(dpacks, targets)
        # save classifier, if necessary
        if cache_file is not None:
            # print('\tsave {}'.format(cache_file))
            joblib.dump(self._learner_su, cache_file)
        return self

    def transform(self, dpack, nonfixed_pairs=None):
        """
        Its main effect is to update the arrays of
        scores for attachment and labelling in `dpack`, for all
        candidate edges where a "same-unit" has been predicted.
        This is a sort of side-effect, so one should be careful
        about it when using this function.
        """

        if dpack.graph is None:
            # SklearnSameUnitClassifier.predict_score requires a
            # weighted datapack
            dpack = self.multiply(dpack)

        # we'll update these copies
        scores_att = np.copy(dpack.graph.attach)
        scores_lbl = np.copy(dpack.graph.label)

        # su_pack, _ = for_attachment_same_unit(dpack, dpack.target)
        su_pack = dpack
        # WIP filter: pass only nonfixed, right-attachment, intra-sentential
        # pairs to the classifier
        ri_pairs = right_intra_idc(su_pack)
        if nonfixed_pairs is not None:
            nf_pairs = list(np.intersect1d(ri_pairs, nonfixed_pairs))
        else:
            nf_pairs = ri_pairs

        if not nf_pairs:
            # no prediction to make (ex: doc with 2 EDUs, 1 sentence each):
            # return (copies of) the original scores
            print('no same-unit prediction where ', su_pack.edus[1].id)
            return dpack

        su_pack = su_pack.selected(nf_pairs)
        # end filter

        scores_pred = self._learner_su.predict_score(su_pack)

        # positive_mask is an array of booleans: True if the
        # corresponding pair has been predicted as "same-unit",
        # False otherwise
        positive_mask = (scores_pred > 0.5
                         if self._learner_su.can_predict_proba
                         else scores_pred > 0)
        # get the absolute indices of pairs for which same-unit has been
        # predicted
        su_pred = np.array(nf_pairs)[positive_mask]
        # WIP 2016-08-25 dump predicted frag EDUs
        if True:
            doc_name = dpack.edus[1].id.rsplit('_', 1)[0]
            print('Predicted same-unit in', doc_name)
            for i, (su_score_pred, pair) in enumerate(zip(
                    scores_pred[positive_mask],
                    [dpack.pairings[i] for i in su_pred]), start=1):
                print('{:.2f}'.format(su_score_pred), pair[0])
                print('    ', pair[1])

            # dump
            fpath_su_pred = doc_name + '.relations.same-unit.deps_pred'
            # assume the first pass is on the whole doc, not a subset like
            # sentence dpack (intra/inter)
            if not os.path.exists(fpath_su_pred):
                frag_edu_pairs_pred = [dpack.pairings[i] for i in su_pred]
                frag_edu_pairs_pred = [(src.id, tgt.id) for src, tgt
                                       in frag_edu_pairs_pred]
                frag_edus_pred = binary_to_nary(
                    'chain', frag_edu_pairs_pred)
                with open(fpath_su_pred, 'wb') as f_out:
                    su_writer = csv.writer(f_out, dialect=csv.excel_tab)
                    for i, frag_edu_members in enumerate(
                            frag_edus_pred, start=1):
                        if len(frag_edu_members) > 2:
                            print(frag_edu_members)
                            raise ValueError('wip wip')

                        doc_name = frag_edu_members[0]
                        frag_edu_id = doc_name + '_frag' + str(i)
                        # FIXME merge same-unit pairs that share an EDU
                        print('should write', frag_edu_id, pair[0].id, pair[1].id)
                        su_writer.writerow([frag_edu_id, pair[0].id, pair[1].id])
        # end dump predicted frag EDUs

        # update the lines of predicted "same-unit" in the matrices of
        # scores for attachment and labels:
        # * attachment: set the score to the predicted score for
        # "same-unit",
        # * label: set the score for "same-unit" to the predicted score,
        # set the scores for other labels to 0.
        scores_att[su_pred] = scores_pred[positive_mask]
        #
        update_lbl = np.zeros(scores_lbl[su_pred].shape, dtype=float)
        su_idx = dpack.label_number(SAME_UNIT)
        update_lbl[:, su_idx] = scores_pred[positive_mask]
        scores_lbl[su_pred] = update_lbl

        # update dpack graph
        graph = dpack.graph.tweak(attach=scores_att,
                                  label=scores_lbl)
        dpack = dpack.set_graph(graph)
        return dpack


class SameUnitJointPipeline(Pipeline):
    """Same-unit preprocessor then JointPipeline.

    Predicted "same-unit" are used to generate new instances.
    The prediction score from the same-unit preprocessor are used to
    overwrite the corresponding attachment and labelling scores in the
    arrays of attachment and labelling scores, before the product of
    scores is computed.

    Parameters
    ----------
    learner_su: SameUnitClassifier

    learner_attach: AttachClassifier

    learner_label: LabelClassifier

    decoder: Decoder

    Notes
    -----
    *Cache keys*

    * attach: attach model path
    * label: label model path
    * su: same-unit model path
    """
    def __init__(self, learner_su, learner_attach, learner_label, decoder):
        if not learner_attach.can_predict_proba:
            raise ValueError('Attachment model does not know how to predict '
                             'probabilities.')
        if not learner_label.can_predict_proba:
            raise ValueError('Relation labelling model does not '
                             'know how to predict probabilities')
        if not learner_su.can_predict_proba:
            raise ValueError('Same-Unit model does not '
                             'know how to predict probabilities')

        steps = [
            ('same-unit weights', SameUnitClassifierWrapper(learner_su)),
            #
            ('attach weights', AttachClassifierWrapper(learner_attach)),
            ('label weights', LabelClassifierWrapper(learner_label)),
            ('attach x best label', AttachTimesBestLabel()),
            ('decoder', decoder)
        ]
        super(SameUnitJointPipeline, self).__init__(steps=steps)


class JointSameUnitPipeline(Pipeline):
    """JointPipeline with an extra step to predict "same-unit".

    The scores of predicted "same-unit" are used to overwrite the
    corresponding attachment and labelling scores in the arrays of
    attachment and labelling scores, before the product of scores is
    computed.

    Parameters
    ----------
    learner_attach: AttachClassifier

    learner_label: LabelClassifier

    learner_su: SameUnitClassifier

    decoder: Decoder

    Notes
    -----
    *Cache keys*

    * attach: attach model path
    * label: label model path
    * su: same-unit model path
    """
    def __init__(self, learner_attach, learner_label, learner_su, decoder):
        if not learner_attach.can_predict_proba:
            raise ValueError('Attachment model does not know how to predict '
                             'probabilities.')
        if not learner_label.can_predict_proba:
            raise ValueError('Relation labelling model does not '
                             'know how to predict probabilities')
        if not learner_su.can_predict_proba:
            raise ValueError('Same-Unit model does not '
                             'know how to predict probabilities')

        steps = [('attach weights', AttachClassifierWrapper(learner_attach)),
                 ('label weights', LabelClassifierWrapper(learner_label)),
                 ('same-unit weights', SameUnitClassifierWrapper(learner_su)),
                 ('attach x best label', AttachTimesBestLabel()),
                 ('decoder', decoder)]
        super(JointSameUnitPipeline, self).__init__(steps=steps)
