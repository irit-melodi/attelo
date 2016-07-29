"""Document-level parsers that first do sentence-level parsing.

An IntraInterParser applies separate parsers on edges within a sentence
and then on edges across sentences.
"""

from __future__ import print_function
from collections import defaultdict, namedtuple

from abc import ABCMeta, abstractmethod
from six import with_metaclass
import numpy as np

from attelo.edu import (FAKE_ROOT_ID, edu_id2num)
from attelo.table import (DataPack,
                          Graph,
                          UNRELATED,
                          idxes_inter,
                          idxes_intra,
                          locate_in_subpacks,
                          grouped_intra_pairings)
from .interface import (Parser)

# pylint: disable=too-few-public-methods


class IntraInterPair(namedtuple("IntraInterPair",
                                "intra inter")):
    """
    Any pair of the same sort of thing, but with one meant
    for intra-sentential decoding, and the other meant for
    intersentential
    """
    def fmap(self, fun):
        """Return the result of applying a function on both intra/inter

        Parameters
        ----------
        fun: `a -> b`

        Returns
        -------
        IntraInterPair(b)
        """
        return IntraInterPair(intra=fun(self.intra),
                              inter=fun(self.inter))

def for_intra(dpack, target):
    """Adapt a datapack to intrasentential decoding.

    An intrasentential datapack is almost identical to its original,
    except that we set the label for each ('ROOT', edu) pairing to
    'ROOT' if that edu is a subgrouping head (if it has no parents other
    than 'ROOT' within its subgrouping).

    This should be done before either `for_labelling` or `for_attachment`

    Returns
    -------
    dpack : DataPack

    target : array(int)

    """
    # map EDUs to subgroup ids ; intra = pairs of EDUs with same subgroup id
    grp = {e.id: e.subgrouping for e in dpack.edus}
    # find all edus that have intra incoming edges (to rule out)
    unrelated = dpack.label_number(UNRELATED)
    intra_tgts = defaultdict(set)
    for i, (edu1, edu2) in enumerate(dpack.pairings):
        if (grp[edu1.id] == grp[edu2.id]
            and target[i] != unrelated):
            # edu2 has an incoming relation => not an (intra) root
            intra_tgts[grp[edu2.id]].add(edu2.id)
    # pick out the (fakeroot, edu) pairs where edu does not have
    # incoming intra edges
    all_heads = [i for i, (edu1, edu2) in enumerate(dpack.pairings)
                 if (edu1.id == FAKE_ROOT_ID
                     and edu2.id not in intra_tgts[grp[edu2.id]])]
    # NEW pick out the original inter-sentential links, for removal
    inter_links = [i for i, (edu1, edu2) in enumerate(dpack.pairings)
                   if (edu1.id != FAKE_ROOT_ID
                       and grp[edu1.id] != grp[edu2.id]
                       and target[i] != unrelated)]
    # 2016-07-29 CDUs
    all_heads_cdu = []
    for i, (du1, du2) in enumerate(dpack.cdu_pairings):
        tgt = (du2.members[0] if isinstance(du2, CDU) else du2)
        if (du1.id == FAKE_ROOT_ID
            and tgt.id not in intra_tgts[grp[tgt.id]]):
            # leftmost member of du2 is an intra root =>
            # keep (ROOT, leftmost member of du2)
            all_heads_cdu.append(i)
    inter_links_cdu = []
    for i, (du1, du2) in enumerate(dpack.cdu_pairings):
        src = (du1.members[0] if isinstance(du1, CDU) else du1)
        tgt = (du2.members[0] if isinstance(du2, CDU) else du2)
        if (src.id != FAKE_ROOT_ID
            and grp[src.id] != grp[tgt.id]
            and cdu_target[i] != unrelated):
            # inter link should be removed
            inter_links_cdu.append(i)
    new_cdu_target = np.copy(dpack.cdu_target)
    new_cdu_target[all_heads_cdu] = dpack.label_number('ROOT')
    new_cdu_target[inter_links_cdu] = unrelated
    # end CDUs


    # update datapack and target accordingly
    new_target = np.copy(dpack.target)
    new_target[all_heads] = dpack.label_number('ROOT')
    new_target[inter_links] = unrelated  # NEW
    # WIP ctarget
    new_ctarget = {grp_name: ctgt
                   for grp_name, ctgt in dpack.ctarget.items()}
    # FIXME replace each ctgt with the list of intra-sentential
    # RST (sub)trees
    # end WIP ctarget
    dpack = DataPack(edus=dpack.edus,
                     pairings=dpack.pairings,
                     data=dpack.data,
                     target=new_target,
                     ctarget=new_ctarget,
                     # 2016-07-28 WIP CDUs
                     cdus=dpack.cdus,
                     cdu_pairings=dpack.cdu_pairings,
                     cdu_data=dpack.cdu_data,
                     cdu_target=new_cdu_target,
                     # end WIP CDUs
                     labels=dpack.labels,
                     vocab=dpack.vocab,
                     graph=dpack.graph)
    target = np.copy(target)
    target[all_heads] = dpack.label_number('ROOT')
    target[inter_links] = unrelated  # NEW
    return dpack, target


def _zip_sentences(func, sent_parses):
    """
    Given a function that combines sentence predictions, and
    given a list of predictions for each sentence (eg. the
    3 best predictions for each sentence), apply that function
    over each list that goes together (eg. on best predictions
    for each sentence, then the second best, then the third
    best).

    We assume that we have the same number of predictions for
    each sentence ::

         zip_sentences(foo, [[s1_1, s1_2, s1_3],
                             [s2_1, s2_2, s2_3]])
         ==
         [foo([s1_1, s2_1]),
          foo([s1_2, s2_2]),
          foo([s1_3, s3_3])]

    Remember that a prediction is itself a set of links

    :type func: [prediction] -> a
    :type sent_parses: [[prediction]]
    :rtype: [a]
    """
    return [func(list(xs)) for xs in zip(*sent_parses)]


def partition_subgroupings(dpack):
    """Partition the pairings of a datapack along (grouping, subgrouping).

    Parameters
    ----------
    dpack : DataPack
        Datapack to partition

    Returns
    -------
    groups : dict from (string, string) to list of integers
        Map each (grouping, subgrouping) to the list of indices of pairings
        within the same subgrouping.

    Notes
    -----
    * (FAKE_ROOT, x) pairings are included in the group defined by
        (grouping(x), subgrouping(x)).
    * This function is a tiny wrapper around
        `attelo.table.grouped_intra_pairings`.
    """
    return grouped_intra_pairings(dpack, include_fake_root=True).values()


class IntraInterParser(with_metaclass(ABCMeta, Parser)):
    """
    Parser that performs attach, direction, and labelling tasks;
    but in two phases:

        1. by separately parsing edges within the same sentence
        2. and then combining the results to form a document

    This is an abstract class

    Notes
    -----
    /Cache keys/: Same as whatever included parsers would use.
    This parser will divide the dictionary into keys that
    have an 'intra:' prefix or not. The intra prefixed keys
    will be passed onto the intrasentential parser (with
    the prefix stripped). The other keys will be passed onto
    the intersentential parser
    """
    def __init__(self, parsers, sel_inter='inter', verbose=False):
        """
        Parameters
        ----------
        parsers : IntraInterPair(Parser)
            Pair of parsers, for intra- and inter-subgrouping.
        sel_inter : {'inter', 'global', 'head_to_head', 'frontier_to_head'}
            Subset of pairings passed to the inter parser:
            * inter: only inter-subgrouping pairings (default value),
            * global: all (intra and inter) pairings,
            * head_to_head: subset of inter pairings between intra heads,
            * frontier_to_head: susbet of inter pairings from EDUs on
            the left or right frontiers of intra subtrees to heads of
            other intra subtrees.
        """
        self._parsers = parsers
        self._sel_inter = sel_inter
        self._verbose = verbose

    @staticmethod
    def _split_cache(cache):
        """
        Returns
        -------
        caches: IntraInterPair(dict(string, filepath))
        """
        if cache is None:
            return IntraInterPair(None, None)

        intra_cache = {k[len('intra:'):]: v for k, v in cache.items()
                       if k.startswith('intra:')}
        inter_cache = {k[len('inter:'):]: v for k, v in cache.items()
                       if k.startswith('inter:')}
        return IntraInterPair(intra=intra_cache,
                              inter=inter_cache)

    @staticmethod
    def _for_intra_fit(dpack, target):
        """Adapt a datapack for intrasentential learning"""
        idxes = idxes_intra(dpack, include_fake_root=True)
        dpack = dpack.selected(idxes)
        target = target[idxes]
        dpack, target = for_intra(dpack, target)
        return dpack, target

    def _for_inter_fit(self, dpack, target):
        """Adapt a datapack for intersentential learning.

        Parameters
        ----------
        dpack : DataPack
            DataPack containing the EDUs and their features.
        target : array(int)
            Ground truth labels.
        sel_inter : {'inter', 'global', 'head_to_head', 'frontier_to_head'}
            See docstring for `__init__`.

        Returns
        -------
        dpack : DataPack
        target : array(int)

        Notes
        -----
        Please be careful that irit_rst_dt.harness.IritHarness.model_paths
        has different suffixes for the various inter-sentential selections,
        e.g. 'inter:attach': ..."{doc,doc_heads,...}-attach"), and
        'attach': ... "attach".
        """
        sel_inter = self._sel_inter

        # 'global': keep all pairings
        if sel_inter == 'global':
            return dpack, target

        if sel_inter == 'inter':
            idxes = idxes_inter(dpack, include_fake_root=True)
        elif sel_inter == 'head_to_head':
            # keep edges between sentential heads (plus the fake root)
            # find all EDUs that have intra incoming edges in gold (to rule
            # out)
            unrelated = dpack.label_number(UNRELATED)
            pairs_true = np.where(target != unrelated)[0]
            pairs_intra = idxes_intra(dpack, include_fake_root=False)
            pairs_intra_true = np.intersect1d(pairs_true, pairs_intra)
            intra_tgts = set(dpack.pairings[i][1].id
                             for i in pairs_intra_true)
            # pick out (fakeroot, head_i) or (head_i, head_j) inter edges
            pairs_inter = idxes_inter(dpack, include_fake_root=True)
            idxes = [i for i in pairs_inter
                     if (dpack.pairings[i][0].id not in intra_tgts and
                         dpack.pairings[i][1].id not in intra_tgts)]
        elif sel_inter == 'frontier_to_head':
            # TODO refactor and optimize !
            # possible helper: attelo.table.grouped_intra_pairings

            # find all EDUs that have intra incoming edges (to rule out)
            unrelated = dpack.label_number(UNRELATED)
            intra_tgts = defaultdict(set)
            # left- and right-most dependents of an EDU, inside subgrouping
            # (key and values are EDU ids i.e. strings)
            lmost_dep = dict()
            rmost_dep = dict()
            for i, (edu1, edu2) in enumerate(dpack.pairings):
                if ((edu1.subgrouping == edu2.subgrouping and
                     target[i] != unrelated)):
                    intra_tgts[edu2.subgrouping].add(edu2.id)
                    # update left- or right-most dependent if relevant:
                    if edu_id2num(edu1.id) < edu_id2num(edu2.id):
                        # right attachment
                        if ((edu1.id not in rmost_dep or
                             (edu_id2num(rmost_dep[edu1.id]) <
                              edu_id2num(edu2.id)))):
                            rmost_dep[edu1.id] = edu2.id
                    else:  # left attachment
                        if ((edu1.id not in lmost_dep or
                             (edu_id2num(edu2.id) <
                              edu_id2num(lmost_dep[edu1.id])))):
                            lmost_dep[edu1.id] = edu2.id
            # use intra_tgts to gather all intra heads
            intra_head_ids = set(e.id for e in dpack.edus
                                 if e.id not in intra_tgts[e.subgrouping])
            # then compute the left and right frontier of each subtree
            intra_lfrontier = set()
            intra_rfrontier = set()
            for head_id in intra_head_ids:
                # left frontier
                lmost_cur = head_id
                while lmost_cur is not None:
                    intra_lfrontier.add(lmost_cur)
                    lmost_cur = lmost_dep.get(lmost_cur, None)
                # right frontier
                rmost_cur = head_id
                while rmost_cur is not None:
                    intra_rfrontier.add(rmost_cur)
                    rmost_cur = rmost_dep.get(rmost_cur, None)
            # pick out (fakeroot or rfrontier, head) right attachments or
            # (lfrontier, head) left attachments
            idxes = [i for i, (edu1, edu2) in enumerate(dpack.pairings)
                     if (edu2.id not in intra_tgts[edu2.subgrouping] and
                         ((edu1.id == FAKE_ROOT_ID or
                           edu1.id in intra_rfrontier) and
                          edu_id2num(edu2.id) > edu_id2num(edu1.id)) or
                         (edu1.id in intra_lfrontier and
                          edu_id2num(edu2.id) < edu_id2num(edu1.id)))]

        # filter dpack and target on these indices
        dpack = dpack.selected(idxes)
        target = target[idxes]
        return dpack, target

    def fit(self, dpacks, targets, cache=None):
        caches = self._split_cache(cache)

        # print('intra.fit')
        if dpacks:
            dpacks_intra, targets_intra = self.dzip(self._for_intra_fit,
                                                    dpacks, targets)
        else:
            dpacks_intra, targets_intra = dpacks, targets

        dpacks_spacks = []
        targets_spacks = []
        for i, dpack_intra in enumerate(dpacks_intra):
            subgrp_idxs = partition_subgroupings(dpack_intra)

            dpack_spacks = [dpack_intra.selected(idxs)
                            for idxs in subgrp_idxs]
            dpacks_spacks.extend(dpack_spacks)

            target_intra = targets_intra[i]
            target_spacks = [target_intra[idxs]
                             for idxs in subgrp_idxs]
            targets_spacks.extend(target_spacks)
        self._parsers.intra.fit(dpacks_spacks, targets_spacks,
                                cache=caches.intra)
        # print('inter.fit')
        if dpacks:
            dpacks_inter, targets_inter = self.dzip(self._for_inter_fit,
                                                    dpacks, targets)
        else:
            dpacks_inter, targets_inter = dpacks, targets

        inter_indices = [idxes_inter(dpack_inter, include_fake_root=True)
                         for dpack_inter in dpacks_inter]
        self._parsers.inter.fit(dpacks_inter, targets_inter,
                                nonfixed_pairs=inter_indices,
                                cache=caches.inter)
        return self

    def transform(self, dpack):
        # intrasentential target links are slightly different
        # in the fakeroot case (this only really matters if we
        # are using an oracle)
        dpack = self.multiply(dpack)

        # call intra parser
        dpack_intra, _ = for_intra(dpack, dpack.target)
        # parse each sentence
        spacks = [dpack_intra.selected(idxs)
                  for idxs in partition_subgroupings(dpack_intra)]
        spacks = [self._parsers.intra.transform(spack)
                  for spack in spacks]

        # call inter parser with intra predictions
        dpack_inter = dpack
        dpack_pred = self._recombine(dpack_inter, spacks)

        return dpack_pred

    @staticmethod
    def _mk_get_lbl(dpack, subpacks):
        """
        Return a function that retrieves the label for an
        item within one of the subpacks, or None if it's
        not present

        Return
        ------
        get_lbl: int -> int or None
        """
        sub_idxes = locate_in_subpacks(dpack, subpacks)

        def get_lbl(i):
            'retrieve lbl if present'
            if sub_idxes[i] is None:
                return None
            else:
                spack, j = sub_idxes[i]
                return spack.graph.prediction[j]
        return get_lbl

    @abstractmethod
    def _recombine(self, dpack, spacks):
        """
        Run the second phase of decoding combining the results
        from the first phase
        """
        return NotImplementedError

    def _fix_intra_edges(self, dpack, spacks):
        """Fix intra-sentential edges for inter-sentential parsing.

        Scores are set to 1.0 for both attachment and labelling, for
        intra-sentential links.

        Parameters
        ----------
        dpack : DataPack
            Original datapack.

        spacks : list of DataPack
            List of intra-sentential datapacks, containing intra-sentential
            predictions.

        Returns
        -------
        dpack_copy : DataPack
            Copy of dpack with attachment and labelling scores updated.

        FIXME
        -----
        [ ] generalize to support non-probabilistic scores
        """
        # NB this code was moved here from SoftParser._recombine()
        # it probably leaves room for improvement, notably speedups
        unrelated_lbl = dpack.label_number(UNRELATED)
        sent_lbl = self._mk_get_lbl(dpack, spacks)

        # tweak intra-sentential attachment and labelling scores
        weights_a = np.copy(dpack.graph.attach)
        weights_l = np.copy(dpack.graph.label)
        for i, (edu1, edu2) in enumerate(dpack.pairings):
            if edu1.id == FAKE_ROOT_ID:
                # don't confuse the inter parser with sentence roots
                continue
            lbl = sent_lbl(i)
            if lbl is not None and lbl != unrelated_lbl:
                weights_a[i] = 1.0
                weights_l[i] = np.zeros(len(dpack.labels))
                weights_l[i, lbl] = 1.0

        # FIXME "legacy" code that used to be in learning.oracle
        # it looks simpler thus better than what precedes, but is it
        # (partly) functionally equivalent in our pipelines?
        if False:  # if _pass_intras:
            # set all intra attachments to 1.0
            intra_pairs = idxes_intra(dpack, include_fake_root=False)
            weights_a[intra_pairs] = 1.0  # replace res with (?)
        # end FIXME

        graph = Graph(prediction=dpack.graph.prediction,
                      attach=weights_a,
                      label=weights_l)
        dpack_copy = dpack.set_graph(graph)
        return dpack_copy

    def _check_intra_edges(self, dpack, spacks):
        """Compare gold and predicted intra-sentential edges.

        Lost and hallucinated intra-sentential edges are printed on stdout.

        Parameters
        ----------
        dpack : DataPack
            Global datapack

        spacks : list of DataPack
            Sentential datapacks
        """
        unrelated_lbl = dpack.label_number(UNRELATED)
        # intra-sentential predictions
        sent_lbl = self._mk_get_lbl(dpack, spacks)

        idxes_intra_pred = [i for i, (edu1, edu2) in enumerate(dpack.pairings)
                            if (edu1.subgrouping == edu2.subgrouping and
                                sent_lbl(i) != unrelated_lbl)]
        idxes_intra_true = [i for i, (edu1, edu2) in enumerate(dpack.pairings)
                            if (edu1.subgrouping == edu2.subgrouping and
                                dpack.target[i] != unrelated_lbl)]
        if set(idxes_intra_true) != set(idxes_intra_pred):
            lost_intra_edges = set(idxes_intra_true) - set(idxes_intra_pred)
            if lost_intra_edges:
                print('Lost intra edges:')
                print([(e1.id, e2.id)
                       for i, (e1, e2) in enumerate(dpack.pairings)
                       if i in lost_intra_edges])
            hall_intra_edges = set(idxes_intra_pred) - set(idxes_intra_true)
            if hall_intra_edges:
                print('Hallucinated intra edges:')
                print([(e1.id, e2.id)
                       for i, (e1, e2) in enumerate(dpack.pairings)
                       if i in hall_intra_edges])


class SentOnlyParser(IntraInterParser):
    """
    Intra/inter parser with no sentence recombination.
    We also chop off any fakeroot connections
    """
    def _recombine(self, dpack, spacks):
        "join sentences by parsing their heads"
        unrelated_lbl = dpack.label_number(UNRELATED)
        sent_lbl = self._mk_get_lbl(dpack, spacks)

        def merged_lbl(i, pair):
            """Doc label where relevant else sentence label.

            Returns
            -------
            lbl: string
                Predicted (intra-sentential) label ; UNRELATED for
                missing values (None) and when EDU1 is the fake root.
            """
            edu1, _ = pair
            lbl = sent_lbl(i)
            # UNRELATED for missing values and edges from the fake root
            if lbl is None or edu1.id == FAKE_ROOT_ID:
                lbl = unrelated_lbl
            return lbl

        # merge results
        prediction = np.fromiter((merged_lbl(i, pair)
                                  for i, pair in enumerate(dpack.pairings)),
                                 dtype=np.dtype(np.int16))
        graph = dpack.graph.tweak(prediction=prediction)
        dpack = dpack.set_graph(graph)
        return dpack


class HeadToHeadParser(IntraInterParser):
    """Intra/inter parser in which sentence recombination consists of
    parsing with only sentence heads.

    TODO
    ----
    [ ] write and integrate an oracle that replaces lost gold edges (from
    non-head to head) with the closest alternative, here moving edges
    up the intra subtrees so they link the (recursive) heads of their
    original nodes.
    """

    def _select_heads(self, dpack, spacks):
        """Return datapack consisting only of links between sentence
        heads and each other or the fakeroot.

        Parameters
        ----------
        dpack : DataPack
            Global datapack

        spacks : list of DataPack
            Datapacks for each sentence including intra predictions

        Returns
        -------
        dpack : DataPack
            dpack restricted to predicted sentence heads and links on them
        """
        # identify sentence heads
        unrelated_lbl = dpack.label_number(UNRELATED)
        sent_lbl = self._mk_get_lbl(dpack, spacks)
        head_ids = [edu2.id for i, (edu1, edu2) in enumerate(dpack.pairings)
                    if (edu1.id == FAKE_ROOT_ID and
                        sent_lbl(i) != unrelated_lbl)]

        # pick out edges where both elements are
        # a sentence head (or the fake root)
        def is_head_or_root(edu):
            'true if an edu is the fake root or a sentence head'
            return edu.id == FAKE_ROOT_ID or edu.id in head_ids

        idxes = [i for i, (e1, e2) in enumerate(dpack.pairings)
                 if (is_head_or_root(e1) and
                     is_head_or_root(e2))]

        if self._verbose:
            # check for lost inter edges
            idxes_er = [i for i, (e1, e2) in enumerate(dpack.pairings)
                        if (e1.subgrouping != e2.subgrouping and
                            dpack.target[i] != unrelated_lbl)]
            if not set(idxes_er).issubset(set(idxes)):
                print('Lost inter indices:')
                print([(e1.id, e2.id)
                       for i, (e1, e2) in enumerate(dpack.pairings)
                       if i in set(idxes_er) - set(idxes)])

        return dpack.selected(idxes)

    def _recombine(self, dpack, spacks):
        "join sentences by parsing their heads"
        unrelated_lbl = dpack.label_number(UNRELATED)
        # intra-sentential predictions
        sent_lbl = self._mk_get_lbl(dpack, spacks)

        if self._verbose:
            # check for lost and hallucinated intra- edges
            self._check_intra_edges(dpack, spacks)

        # call inter-sentential parser
        dpack_inter = self._select_heads(dpack, spacks)
        has_inter = len(dpack_inter) > 0
        if has_inter:
            dpack_inter = self._parsers.inter.transform(dpack_inter)

        doc_lbl = self._mk_get_lbl(dpack, [dpack_inter])

        def merged_lbl(i):
            """Doc label where relevant else sentence label.

            Returns
            -------
            lbl:  string
                Predicted document-level label, else sentence-level
                label ; UNRELATED for missing values.
            """
            lbl = doc_lbl(i) if has_inter else None
            # missing document-level prediction: use sentence-level
            # prediction
            if lbl is None:
                lbl = sent_lbl(i)
            # fallback: it may have fallen through the cracks
            # (ie. may be neither in a sentence be a head)
            if lbl is None:
                lbl = unrelated_lbl
            return lbl

        # merge results
        prediction = np.fromiter((merged_lbl(i)
                                  for i in range(len(dpack))),
                                 dtype=np.dtype(np.int16))
        graph = dpack.graph.tweak(prediction=prediction)
        dpack = dpack.set_graph(graph)

        if self._verbose:
            # check for hallucinated and lost inter edges
            inter_edges_pred = [(edu1.id, edu2.id, sent_lbl(i))
                                for i, (edu1, edu2) in enumerate(dpack.pairings)
                                if (edu1.subgrouping != edu2.subgrouping and
                                    merged_lbl(i) != unrelated_lbl)]
            inter_edges_true = [(edu1.id, edu2.id, dpack.target[i])
                                for i, (edu1, edu2) in enumerate(dpack.pairings)
                                if (edu1.subgrouping != edu2.subgrouping and
                                    dpack.target[i] != unrelated_lbl)]
            if set(inter_edges_true) != set(inter_edges_pred):
                print('Lost inter edges: {}'.format(sorted(set(inter_edges_true) - set(inter_edges_pred))))
                print()
                print('Hallucinated inter edges: {}'.format(sorted(set(inter_edges_pred) - set(inter_edges_true))))

        return dpack


class FrontierToHeadParser(IntraInterParser):
    """Intra/inter parser in which sentence recombination consists of
    parsing with edges from the frontier of sentential subtree to sentence
    head.

    TODO
    ----
    [ ] write and integrate an oracle that replaces lost gold edges (from
    non-head to head) with the closest alternative ; here this probably
    happens on leaky sentences and I still have to figure out what an
    oracle should look like.
    """

    def _select_frontiers(self, dpack, spacks):
        """Restrict dpack to edges necessary for inter parsing.

        This restricts dpack to two types of edges:
        * predicted edges on the left and right frontier of each sentential
        tree,
        * potential edges from the predicted (left and right) frontiers of
        sentential subtrees to heads of the other sentential subtrees (on
        the left and right, respectively).

        `_for_inter_fit()` keeps only the second type of edges.

        Parameters
        ----------
        dpack : DataPack
            Global datapack.

        spacks : list of DataPack
            Datapacks for each sentence including intra predictions.

        Returns
        -------
        dpack : DataPack
            Restricted dpack.
        """
        # identify sentence heads
        unrelated_lbl = dpack.label_number(UNRELATED)
        sent_lbl = self._mk_get_lbl(dpack, spacks)
        head_ids = [edu2.id for i, (edu1, edu2) in enumerate(dpack.pairings)
                    if (edu1.id == FAKE_ROOT_ID and
                        sent_lbl(i) != unrelated_lbl)]

        # compute left and right frontiers
        # * first, gather left- and right-most predicted dependents
        lmost_dep = dict()
        rmost_dep = dict()
        for i, (edu1, edu2) in enumerate(dpack.pairings):
            if ((edu1.subgrouping == edu2.subgrouping and
                 sent_lbl(i) != unrelated_lbl)):
                if edu_id2num(edu1.id) < edu_id2num(edu2.id):  # right attach
                    if ((edu1.id not in rmost_dep or
                         (edu1.id in rmost_dep and
                          edu_id2num(edu2.id) > edu_id2num(rmost_dep[edu1.id])))):
                        rmost_dep[edu1.id] = edu2.id
                else:  # left attachment
                    if ((edu1.id not in lmost_dep or
                         (edu1.id in lmost_dep and
                          edu_id2num(edu2.id) < edu_id2num(lmost_dep[edu1.id])))):
                        lmost_dep[edu1.id] = edu2.id
        # * finally, we can compute the left and right frontier of each
        # sentential tree
        intra_lfrontier = set()
        intra_rfrontier = set()
        for head_id in head_ids:
            lmost_cur = head_id
            while lmost_cur is not None:
                intra_lfrontier.add(lmost_cur)
                lmost_cur = lmost_dep.get(lmost_cur, None)
            rmost_cur = head_id
            while rmost_cur is not None:
                intra_rfrontier.add(rmost_cur)
                rmost_cur = rmost_dep.get(rmost_cur, None)
        # end WIP
        # pick out (fakeroot or rfrontier, head) right attachments or
        # (lfrontier, head) left attachments
        def frontier_to_head_edge(edu1, edu2):
            """True if edu1 is a frontier node with access to edu2 as an
            intra head
            """
            return (edu2.id in head_ids and
                    (((edu1.id == FAKE_ROOT_ID or
                       edu1.id in intra_rfrontier) and
                      edu_id2num(edu2.id) > edu_id2num(edu1.id)) or
                     (edu1.id in intra_lfrontier and
                      edu_id2num(edu2.id) < edu_id2num(edu1.id))))

        def same_frontier_edge(edu1, edu2):
            """True if edu1 and edu2 are linked and on the same intra
            frontier
            """
            return (edu1.subgrouping == edu2.subgrouping and
                    ((edu1.id in intra_rfrontier and
                      edu2.id in intra_rfrontier and
                      sent_lbl(i) != unrelated_lbl) or
                     (edu1.id in intra_lfrontier and
                      edu2.id in intra_lfrontier and
                      sent_lbl(i) != unrelated_lbl)))

        idxes = [i for i, (edu1, edu2) in enumerate(dpack.pairings)
                 if (frontier_to_head_edge(edu1, edu2) or
                     same_frontier_edge(edu1, edu2))]

        if self._verbose:
            # check for lost inter edges
            idxes_er = [i for i, (e1, e2) in enumerate(dpack.pairings)
                        if (e1.subgrouping != e2.subgrouping and
                            dpack.target[i] != unrelated_lbl)]
            if not set(idxes_er).issubset(set(idxes)):
                print('Lost inter indices:')
                print([(e1.id, e2.id)
                       for i, (e1, e2) in enumerate(dpack.pairings)
                       if i in set(idxes_er) - set(idxes)])

        dpack_frontier = dpack.selected(idxes)
        return dpack_frontier

    def _recombine(self, dpack, spacks):
        """Parse a document using partial parses for each subgroup.

        The current implementation behaves like the SoftParser, requiring
        a global model to score both the (almost fixed) intra edges and the
        inter edges.

        Parameters
        ----------
        dpack : DataPack
            Datapack for the whole document

        spacks : list of DataPack
            List of datapacks, one per subgroup (sentence).

        Returns
        -------
        dpack : DataPack
            Datapack for the whole document, filled with a parse.
        """
        unrelated_lbl = dpack.label_number(UNRELATED)
        # intra-sentential predictions
        sent_lbl = self._mk_get_lbl(dpack, spacks)

        if self._verbose:
            # check for lost and hallucinated intra- edges
            print('>>> check intra 1 >>>')
            self._check_intra_edges(dpack, spacks)
            print('<<< end check intra 1 <<<')

        # fix intra-sentential decisions before the inter-sentential phase
        dpack = self._fix_intra_edges(dpack, spacks)

        # call inter-sentential parser
        dpack_inter = self._select_frontiers(dpack, spacks)
        has_inter = len(dpack_inter) > 0
        if has_inter:
            # collect indices of inter pairings in dpack_inter
            # so we can instruct the inter parser to keep its nose
            # out of intra stuff
            inter_indices = idxes_inter(dpack_inter, include_fake_root=True)
            dpack_inter = self._parsers.inter.transform(
                dpack_inter, nonfixed_pairs=inter_indices)

        doc_lbl = self._mk_get_lbl(dpack, [dpack_inter])

        def merged_lbl(i):
            """Doc label where relevant else sentence label.

            Returns
            -------
            lbl:  string
                Predicted document-level label, else sentence-level
                label ; UNRELATED for missing values.
            """
            lbl = doc_lbl(i) if has_inter else None
            # missing document-level prediction: use sentence-level
            # prediction
            if lbl is None:
                lbl = sent_lbl(i)
            # fallback: it may have fallen through the cracks
            # (ie. may be neither in a sentence be a head)
            if lbl is None:
                lbl = unrelated_lbl
            return lbl

        # merge results
        prediction = np.fromiter((merged_lbl(i)
                                  for i in range(len(dpack))),
                                 dtype=np.dtype(np.int16))
        graph = dpack.graph.tweak(prediction=prediction)
        dpack = dpack.set_graph(graph)

        if self._verbose:
            # 2nd check for lost and hallucinated intra- edges
            print('>>> check intra 2 >>>')
            self._check_intra_edges(dpack, spacks)
            print('<<< end check intra 2 <<<')
            # check for lost and hallucinated inter- edges
            # TODO turn into _check_inter_edges
            inter_edges_pred = [(edu1.id, edu2.id, merged_lbl(i))
                                for i, (edu1, edu2)
                                in enumerate(dpack.pairings)
                                if (edu1.subgrouping != edu2.subgrouping and
                                    merged_lbl(i) != unrelated_lbl)]
            inter_edges_true = [(edu1.id, edu2.id, dpack.target[i])
                                for i, (edu1, edu2)
                                in enumerate(dpack.pairings)
                                if (edu1.subgrouping != edu2.subgrouping and
                                    dpack.target[i] != unrelated_lbl)]
            if set(inter_edges_true) != set(inter_edges_pred):
                print('Lost inter edges: {}'.format(
                    sorted(set(inter_edges_true) - set(inter_edges_pred))))
                print()
                print('Hallucinated inter edges: {}'.format(
                    sorted(set(inter_edges_pred) - set(inter_edges_true))))

        return dpack
# end WIP


class SoftParser(IntraInterParser):
    """
    Intra/inter parser in which sentence recombination consists of

    1. passing intra-sentential edges through but
    2. marking 1.0 attachment probabilities if they are attached
       and 1.0 label probabilities on the resulting edge

    Notes
    -----
    In its current implementation, this parser needs a global model,
    i.e. one fit on the whole dataset, so that it can correctly score
    intra-sentential edges.
    Different, alternative implementations could probably solve or work
    around this.
    """
    def _recombine(self, dpack, spacks):
        "soft decoding - pass sentence edges through the prob dist"
        dpack = self._fix_intra_edges(dpack, spacks)
        # call the inter parser on the updated dpack
        dpack = self._parsers.inter.transform(dpack)
        return dpack
