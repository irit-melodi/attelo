"""
Intrasentential decoding mixin

Utitilies that would allow us to extend a decoder with
the ability to do decoding separately for each EDU
subgrouping and then join the results together.
"""

from __future__ import print_function
from collections import defaultdict, namedtuple
import copy

from ..edu import (FAKE_ROOT_ID)
from ..table import (UNRELATED)
from ..util import (ArgparserEnum, concat_i, concat_l)
from .interface import (LinkPack)

# pylint: disable=too-few-public-methods


class IntraStrategy(ArgparserEnum):
    """
    Intrasentential decoding strategy

        * only: (mostly for debugging), do not attach
          sentences together at all
        * heads: attach heads of sentences to each other
        * soft: pass all nodes through to decoder, but
          assign intrasentential links from the sentence
          level decoder a probability of 1
    """
    only = 1
    heads = 2
    soft = 3


class IntraInterPair(namedtuple("IntraInterPair",
                                "intra inter")):
    """
    Any pair of the same sort of thing, but with one meant
    for intra-sentential decoding, and the other meant for
    intersentential
    """
    pass


def _roots(edge_list):
    """
    Given a list of edges from a sentence parse, return those
    which the fake root points to.

    The targets would be designated as the root of the tree.
    If there is more than one, you should by rights select
    one of them as being the actual root
    """
    return frozenset(e2 for e1, e2, _ in edge_list if
                     e1 == FAKE_ROOT_ID)


def _rootless(edge_list):
    """
    Given a list of edges from a sentence parse, return those
    which are not links from the fake root
    """
    return [(e1, e2, r) for e1, e2, r in edge_list if
            e1 != FAKE_ROOT_ID]


def _zip_sentences(func, sent_parses):
    """
    Given a function that combines sentence predictions, and a
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


def partition_subgroupings(lpack):
    """
    Return an iterable of link packs, each pack consisting of
    pairings within the same subgrouping
    """
    sg_indices = defaultdict(list)
    for i, pair in enumerate(lpack.pairings):
        edu2 = pair[1]
        key = edu2.grouping, edu2.subgrouping
        sg_indices[key].append(i)
    for idxs in sg_indices.values():
        yield lpack.selected(idxs)


class IntraInterDecoder(object):
    """
    Augment a decoder with the ability to do separate decoding
    on sentences and then combine the results.

    Note that the decoder will have to trigger this manually
    by calling `self.decode_intra` when it deems that it would
    appropriate to do so (ie when `self.is_intrasentential()`)
    """
    def __init__(self, decoder, strategy):
        self._decoder = decoder
        self._strategy = strategy
        self._debug = True

    def decode_sentence(self, lpack):
        """
        Run the inner decoder on a single sentence
        """
        return self._decoder.decode(lpack)

    def decode_document(self, lpack, sent_parses):
        """
        Run the inner decoder on the "chunks" resulting from applying
        :py:func:`decode_sentence` on each of the sentences
        """
        def decode_head(sent_edges):
            "attach heads of sentences"
            heads = concat_l(_roots(e) for e in sent_edges)
            blinks = concat_l(_rootless(e) for e in sent_edges)
            is_head_or_root = lambda x: x.id in heads or x.id == FAKE_ROOT_ID
            # FIXME: isn't this a little bit too lax?
            # we surely don't want ALL the FAKEROOT links; just those that
            # involve sentence heads?
            idxes = [i for i, (e1, e2) in enumerate(lpack.pairings)
                     if is_head_or_root(e1) or is_head_or_root(e2)]
            rlinks = self._decoder.decode(lpack.selected(idxes))
            return rlinks + blinks

        def decode_soft(sent_edges):
            "soft decoding - pass sentence edges through the prob dist"
            intra_links = {(e1, e2) for e1, e2, rel in
                           concat_i(_rootless(e) for e in sent_edges)
                           if rel != UNRELATED}
            scores_ad = copy.copy(lpack.scores_ad)
            for i, pair in lpack.pairings:
                if pair in intra_links:
                    scores_ad[i] = 1.0
            doc_lpack = LinkPack(edus=lpack.edus,
                                 labels=lpack.labels,
                                 pairings=lpack.pairings,
                                 scores_ad=scores_ad,
                                 scores_l=lpack.scores_l)
            return self._decoder.decode(doc_lpack)

        if self._strategy == IntraStrategy.only:
            return _zip_sentences(concat_l, sent_parses)
        elif self._strategy == IntraStrategy.heads:
            return concat_l(_zip_sentences(decode_head, sent_parses))
        elif self._strategy == IntraStrategy.soft:
            return concat_l(_zip_sentences(decode_soft, sent_parses))
