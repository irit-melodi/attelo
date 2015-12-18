"""
An InterInterParser applies separate parsers on edges
within a sentence, and then on edges across sentences
"""

from __future__ import print_function
from collections import defaultdict, namedtuple

from abc import ABCMeta, abstractmethod
from six import with_metaclass
import numpy as np

from attelo.decoding.util import simple_candidates  # DEBUG
from attelo.edu import (FAKE_ROOT_ID)
from attelo.table import (DataPack,
                          Graph,
                          UNRELATED,
                          idxes_inter,
                          idxes_intra,
                          locate_in_subpacks)
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
    # find all edus that have intra incoming edges (to rule out)
    unrelated = dpack.label_number(UNRELATED)
    intra_tgts = defaultdict(set)
    for i, (edu1, edu2) in enumerate(dpack.pairings):
        if ((edu1.subgrouping == edu2.subgrouping and
             target[i] != unrelated)):
            intra_tgts[edu2.subgrouping].add(edu2.id)
    # pick out the (fakeroot, edu) pairs where edu does not have
    # incoming intra edges
    all_heads = [i for i, (edu1, edu2) in enumerate(dpack.pairings)
                 if (edu1.id == FAKE_ROOT_ID and
                     edu2.id not in intra_tgts[edu2.subgrouping])]
    # NEW pick out the original inter-sentential links, for removal
    inter_links = [i for i, (edu1, edu2) in enumerate(dpack.pairings)
                   if (edu1.id != FAKE_ROOT_ID and
                       edu1.subgrouping != edu2.subgrouping and
                       target[i] != unrelated)]

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
    """
    Return a list of lists of indices, each nested list consisting of
    the indices of pairings within the same subgrouping
    """
    sg_indices = defaultdict(list)
    for i, (edu1, edu2) in enumerate(dpack.pairings):
        key1 = edu1.grouping, edu1.subgrouping
        key2 = edu2.grouping, edu2.subgrouping
        if edu1.id == FAKE_ROOT_ID or key1 == key2:
            sg_indices[key2].append(i)
    return sg_indices.values()


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
    def __init__(self, parsers):
        """
        Parameters
        ----------
        parsers: IntraInterPair(Parser)
        """
        self._parsers = parsers

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

    @staticmethod
    def _for_inter_fit(dpack, target):
        """Adapt a datapack for intersentential learning"""
        idxes = idxes_inter(dpack, include_fake_root=True)
        dpack = dpack.selected(idxes)
        target = target[idxes]
        return dpack, target

    def fit(self, dpacks, targets, cache=None):
        caches = self._split_cache(cache)
        if dpacks:
            dpacks_intra, targets_intra = self.dzip(self._for_intra_fit,
                                                    dpacks, targets)
            dpacks_inter, targets_inter = self.dzip(self._for_inter_fit,
                                                    dpacks, targets)
        else:
            dpacks_intra, targets_intra = dpacks, targets
            dpacks_inter, targets_inter = dpacks, targets

        # print('intra.fit')
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
        self._parsers.inter.fit(dpacks_inter, targets_inter,
                                cache=caches.inter)
        return self

    def transform(self, dpack):
        # intrasentential target links are slightly different
        # in the fakeroot case (this only really matters if we
        # are using an oracle)
        dpack = self.multiply(dpack)
        dpack_intra, _ = for_intra(dpack, dpack.target)
        dpack_inter = dpack
        # parse each sentence
        spacks = [dpack_intra.selected(idxs)
                  for idxs in partition_subgroupings(dpack_intra)]
        spacks = [self._parsers.intra.transform(spack)
                  for spack in spacks]
        # call inter parser with intra predictions
        return self._recombine(dpack_inter, spacks)

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

    @staticmethod
    def _for_inter_fit(dpack, target):
        """Adapt a datapack for intersentential learning.

        This is a custom version of `IntraInterParser._for_inter_fit`
        to restrict instances to edges between sentential heads (plus
        the fake root).

        Parameters
        ----------
        dpack: DataPack
        target: array(int)

        Returns
        -------
        dpack: DataPack
        target: array(int)

        Notes
        -----
        Preliminary experiments indicate this works well in practice.
        Please be careful that irit_rst_dt.harness.IritHarness.model_paths
        attributes different suffixes to global and inter-sentential models,
        e.g. 'inter:{attach,label}': ..."doc-{attach,relate}"), and
        '{attach,label}': ...{attach,relate}".
        """
        # could be a function like
        # `idxes_inter_iheads(dpack, target, include_fake_root=True)`
        # but existing functions in attelo.table only depend on dpack
        # and ignore target (necessary here)
        # find all EDUs that have intra incoming edges (to rule out)
        unrelated = dpack.label_number(UNRELATED)
        intra_tgts = defaultdict(set)
        for i, (edu1, edu2) in enumerate(dpack.pairings):
            if ((edu1.subgrouping == edu2.subgrouping and
                 target[i] != unrelated)):
                intra_tgts[edu2.subgrouping].add(edu2.id)
        # pick out (fakeroot, head_i) or (head_i, head_j) edges
        idxes = [i for i, (edu1, edu2) in enumerate(dpack.pairings)
                 if ((edu1.id == FAKE_ROOT_ID or
                      edu1.id not in intra_tgts[edu1.subgrouping]) and
                     edu2.id not in intra_tgts[edu2.subgrouping])]
        # end idxes_inter_iheads

        dpack = dpack.selected(idxes)
        target = target[idxes]
        return dpack, target

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
        # DEBUG
        idxes_inter = [i for i, (e1, e2) in enumerate(dpack.pairings)
                       if (e1.subgrouping != e2.subgrouping and
                           dpack.target[i] != unrelated_lbl)]
        if not set(idxes_inter).issubset(set(idxes)):
            print('Lost inter indices:')
            print([(e1.id, e2.id) for i, (e1, e2) in enumerate(dpack.pairings)
                   if i in set(idxes_inter) - set(idxes)])
        # end DEBUG
        return dpack.selected(idxes)

    def _recombine(self, dpack, spacks):
        "join sentences by parsing their heads"
        unrelated_lbl = dpack.label_number(UNRELATED)
        # intra-sentential predictions
        sent_lbl = self._mk_get_lbl(dpack, spacks)

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

        # DEBUG
        if False:  # TODO re-enable to debug
            # check inter edges
            inter_edges_pred = [(edu1.id, edu2.id, sent_lbl(i))
                                for i, (edu1, edu2) in enumerate(dpack.pairings)
                                if (edu1.subgrouping != edu2.subgrouping and
                                    merged_lbl(i) != unrelated_lbl)]
            inter_edges_true = [(edu1.id, edu2.id, dpack.target[i])
                                for i, (edu1, edu2) in enumerate(dpack.pairings)
                                if (edu1.subgrouping != edu2.subgrouping and
                                    dpack.target[i] != unrelated_lbl)]
            if set(inter_edges_true) != set(inter_edges_pred):
                print('T&P\t{}'.format(sorted(set(inter_edges_true) & set(inter_edges_pred))))
                print()
                print('T-P\t{}'.format(sorted(set(inter_edges_true) - set(inter_edges_pred))))
                print()
                print('P-T\t{}'.format(sorted(set(inter_edges_pred) - set(inter_edges_true))))
                raise ValueError('Lost inter edges')
                assert set(inter_edges_true) == set(inter_edges_pred)
        # end DEBUG
        return dpack


# WIP

# small helper
def edu_id2num(edu_id):
    """Get the number of an EDU"""
    edu_num = (int(edu_id.rsplit('_', 1)[1])
               if edu_id != FAKE_ROOT_ID
               else 0)
    return edu_num


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

    @staticmethod
    def _for_inter_fit(dpack, target):
        """Adapt a datapack for intersentential learning.

        This is a custom version of `IntraInterParser._for_inter_fit`
        to restrict instances to left (resp. right) edges from the left
        (resp. right) frontier to sentential heads.

        Parameters
        ----------
        dpack : DataPack
            DataPack containing the EDUs and their features.

        target : array(int)
            Grount truth labels.

        Returns
        -------
        dpack : DataPack
            Ibid.

        target : array(int)
            Ibid.

        Notes
        -----
        Preliminary experiments indicate this works well in practice.
        Please check that irit_rst_dt.harness.IritHarness.model_paths does
        attribute different suffixes to global and inter-sentential models,
        e.g. 'inter:{attach,label}': ..."doc-{attach,relate}"), and
        '{attach,label}': ...{attach,relate}".
        """
        # could be a function like
        # `idxes_inter_iheads(dpack, target, include_fake_root=True)`
        # but existing functions in attelo.table only depend on dpack
        # and ignore target (necessary here)
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
                # WIP
                # update left- or right-most dependent if relevant
                if edu_id2num(edu1.id) < edu_id2num(edu2.id):  # right attachment
                    if ((edu1.id not in rmost_dep or
                         (edu1.id in rmost_dep and
                          edu_id2num(edu2.id) > edu_id2num(rmost_dep[edu1.id])))):
                        rmost_dep[edu1.id] = edu2.id
                else:  # left attachment
                    if ((edu1.id not in lmost_dep or
                         (edu1.id in lmost_dep and
                          edu_id2num(edu2.id) < edu_id2num(lmost_dep[edu1.id])))):
                        lmost_dep[edu1.id] = edu2.id
                # end WIP
        # WIP
        # use intra_tgts to gather all intra heads
        intra_head_ids = set()
        for e in dpack.edus:
            if e.id not in intra_tgts[e.subgrouping]:
                intra_head_ids.add(e.id)
        # then compute the left and right frontier of each subtree
        intra_lfrontier = set()
        intra_rfrontier = set()
        for head_id in intra_head_ids:
            lmost_cur = head_id
            while lmost_cur is not None:
                intra_lfrontier.add(lmost_cur)
                lmost_cur = lmost_dep.get(lmost_cur, None)
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
        # end idxes_inter_iheads

        dpack = dpack.selected(idxes)
        target = target[idxes]
        return dpack, target

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
        # DEBUG
        idxes_inter = [i for i, (e1, e2) in enumerate(dpack.pairings)
                       if (e1.subgrouping != e2.subgrouping and
                           dpack.target[i] != unrelated_lbl)]
        if not set(idxes_inter).issubset(set(idxes)):
            print('Lost inter indices:')
            print([(e1.id, e2.id) for i, (e1, e2) in enumerate(dpack.pairings)
                   if i in set(idxes_inter) - set(idxes)])
        # end DEBUG
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

        # call inter-sentential parser
        # DEBUG
        idxes_intra_pred = [i for i, (edu1, edu2) in enumerate(dpack.pairings)
                            if (edu1.subgrouping == edu2.subgrouping and
                                sent_lbl(i) != unrelated_lbl)]
        idxes_intra_true = [i for i, (edu1, edu2) in enumerate(dpack.pairings)
                            if (edu1.subgrouping == edu2.subgrouping and
                                dpack.target[i] != unrelated_lbl)]
        if set(idxes_intra_true) != set(idxes_intra_pred):
            print('Lost intra edges:')
            print([(e1.id, e2.id)
                   for i, (e1, e2) in enumerate(dpack.pairings)
                   if i in set(idxes_intra_true) - set(idxes_intra_pred)])
            print('Hallucinated intra edges:')
            print([(e1.id, e2.id)
                   for i, (e1, e2) in enumerate(dpack.pairings)
                   if i in set(idxes_intra_pred) - set(idxes_intra_true)])
            # raise ValueError('Stop here bro')
        # end DEBUG

        dpack_inter = self._select_frontiers(dpack, spacks)
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
        # DEBUG
        # check intra edges
        intra_edges_pred = [(edu1.id, edu2.id, sent_lbl(i))
                            for i, (edu1, edu2) in enumerate(dpack.pairings)
                            if (edu1.subgrouping == edu2.subgrouping and
                                sent_lbl(i) != unrelated_lbl)]
        intra_edges_true = [(edu1.id, edu2.id, dpack.target[i])
                            for i, (edu1, edu2) in enumerate(dpack.pairings)
                            if (edu1.subgrouping == edu2.subgrouping and
                                dpack.target[i] != unrelated_lbl)]
        assert set(intra_edges_true) == set(intra_edges_pred)
        # DEBUG
        if False:  # TODO re-enable to debug
            # check inter edges
            inter_edges_pred = [(edu1.id, edu2.id, sent_lbl(i))
                                for i, (edu1, edu2) in enumerate(dpack.pairings)
                                if (edu1.subgrouping != edu2.subgrouping and
                                    merged_lbl(i) != unrelated_lbl)]
            inter_edges_true = [(edu1.id, edu2.id, dpack.target[i])
                                for i, (edu1, edu2) in enumerate(dpack.pairings)
                                if (edu1.subgrouping != edu2.subgrouping and
                                    dpack.target[i] != unrelated_lbl)]
            if set(inter_edges_true) != set(inter_edges_pred):
                print('T&P\t{}'.format(sorted(set(inter_edges_true) & set(inter_edges_pred))))
                print()
                print('T-P\t{}'.format(sorted(set(inter_edges_true) - set(inter_edges_pred))))
                print()
                print('P-T\t{}'.format(sorted(set(inter_edges_pred) - set(inter_edges_true))))
                raise ValueError('Lost inter edges')
                assert set(inter_edges_true) == set(inter_edges_pred)
        # end DEBUG
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
        unrelated_lbl = dpack.label_number(UNRELATED)
        sent_lbl = self._mk_get_lbl(dpack, spacks)

        weights_a = np.copy(dpack.graph.attach)
        weights_l = np.copy(dpack.graph.label)
        for i, (edu1, _) in enumerate(dpack.pairings):
            if edu1.id == FAKE_ROOT_ID:
                # don't confuse the inter parser with sentence roots
                continue
            lbl = sent_lbl(i)
            if lbl is not None and lbl != unrelated_lbl:
                weights_a[i] = 1.0
                weights_l[i] = np.zeros(len(dpack.labels))
                weights_l[i, lbl] = 1.0

        graph = Graph(prediction=dpack.graph.prediction,
                      attach=weights_a,
                      label=weights_l)
        dpack = dpack.set_graph(graph)
        # call the inter parser on the updated dpack
        dpack = self._parsers.inter.transform(dpack)
        return dpack
