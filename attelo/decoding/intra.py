"""
Intrasentential decoding mixin

Utitilies that would allow us to extend a decoder with
the ability to do decoding separately for each EDU
subgrouping and then join the results together.
"""

from __future__ import print_function
from collections import namedtuple
import sys

from ..edu import (FAKE_ROOT_ID)
from ..util import (ArgparserEnum, concat_l)
from .util import (get_sorted_edus,
                   subgroupings)

# pylint: disable=too-few-public-methods


class IntraStrategy(ArgparserEnum):
    """
    Intrasentential decoding strategy

        * only: (mostly for debugging), do not attach
          sentences together at all
        * heads: attach heads of sentences to each other
    """
    only = 1
    heads = 2


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
    # pylint: disable=star-args
    return [func(list(xs)) for xs in zip(*sent_parses)]
    # pylint: enable=star-args


def select_subgrouping(prob_distrib, subg):
    """
    Return elements from a probability distribution which belong
    to a subgrouping.

    Note that e silently ignore any EDU pairings which are not in
    the same subgrouping
    """

    def is_in_group(edu):
        "true if an edu is the subgrouping"
        return edu.subgrouping == subg or edu.id == FAKE_ROOT_ID

    return [(e1, e2, prob, rel) for (e1, e2, prob, rel) in prob_distrib
            if is_in_group(e1) and is_in_group(e2)]


def _select_ids(prob_distrib, edu_ids):
    """
    Return elements from a probability distribution in which both
    the source and target have ids in the whitelist
    """
    def is_in_whitelist(edu):
        "true if an edu is the subgrouping"
        return edu.id in edu_ids

    return [(e1, e2, prob, rel) for (e1, e2, prob, rel) in prob_distrib
            if is_in_whitelist(e1) or is_in_whitelist(e2)]


def _select_intra(prob_distrib, sources, targets):
    """
    Return elements from a probability distribution in which

    * the source comes from the list of allowed sources
    * the target comes from the list of allowed targets
    * the source and target are not the same node

    :type sources: [string]
    :type targets: [string]
    """
    return [(e1, e2, prob, rel) for (e1, e2, prob, rel) in prob_distrib
            if e1 in sources and e2 in targets and e1 != e2]


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

    def decode_sentence(self, prob_distrib):
        """
        Run the inner decoder on a single sentence
        """
        return self._decoder.decode(prob_distrib)

    def decode_document(self, prob_distrib, sent_parses):
        """
        Run the inner decoder on the "chunks" resulting from applying
        :py:func:`decode_sentence` on each of the sentences
        """
        def decode_head(sent_edges):
            "attach heads of sentences"
            heads = concat_l(_roots(e) for e in sent_edges)
            blinks = concat_l(_rootless(e) for e in sent_edges)
            doc_distrib = _select_ids(prob_distrib, heads + [FAKE_ROOT_ID])
            rlinks = self._decoder.decode(doc_distrib)
            return rlinks + blinks

        if self._strategy == IntraStrategy.only:
            return _zip_sentences(concat_l, sent_parses)
        elif self._strategy == IntraStrategy.heads:
            return concat_l(_zip_sentences(decode_head, sent_parses))


