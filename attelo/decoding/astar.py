#!/bin/env python
# -*- coding: utf-8 -*-

"""
module for building discourse graphs from probability distribution and
respecting some constraints, using
Astar heuristics based search and variants (beam, b&b)

TODO: unlabelled evaluation seems to bug on RF decoding (relation is of type orange.value
-> go see in decoding.py)


"""

from __future__ import print_function
from argparse import ArgumentTypeError
import copy
import math
import sys
import numpy
from collections import defaultdict, namedtuple
from enum import Enum

from attelo.optimisation.astar import State, Search, BeamSearch
from .interface import Decoder
from .util import get_sorted_edus, get_prob_map

# pylint: disable=too-few-public-methods

_CLASS_SCHEMES = {
    "subord_coord": {
        "subord": frozenset(["elaboration",
                             "e-elab",
                             "attribution",
                             "comment",
                             "flashback",
                             "explanation",
                             "alternation"]),
        "coord": frozenset(["continuation",
                            "parallel",
                            "contrast",
                            "temploc",
                            "frame",
                            "narration",
                            "conditional",
                            "result",
                            "goal",
                            "background"]),
        "NONE": frozenset(["null", "unknown", "NONE"])
        },
    # four class +/- closer to PDTB
    "pdtb": {
        "contingency": frozenset(["explanation",
                                  "conditional",
                                  "result",
                                  "goal",
                                  "background"]),
        "temporal": frozenset(["temploc", "narration", "flashback"]),
        "comparison": frozenset(["parallel", "contrast"]),
        "expansion": frozenset(["frame",
                                "elaboration",
                                "e-elab",
                                "attribution",
                                "continuation",
                                "comment",
                                "alternation"]),
        "error": frozenset(["null", "unknown", "NONE"])
        },
    # our own hierarchy
    "minsdrt": {
        "structural": frozenset(["parallel", "contrast", "alternation", "conditional"]),
        "sequence": frozenset(["result", "narration", "continuation"]),
        "expansion": frozenset(["frame",
                                "elaboration",
                                "e-elab",
                                "attribution",
                                "comment",
                                "explanation"]),
        "temporal": frozenset(["temploc",
                               "goal",
                               "flashback",
                               "background"]),
        "error": frozenset(["null", "unknown", "NONE"])
    }
}


# there's a bit more below
SUBORD_COORD = {"structural": "coord",
                "sequence": "coord",
                "expansion": "subord",
                "temporal": "subord",
                "error": "subord"}

def _flesh_out_subord(scheme):
    'assign coord/subord to relations in our various schemes'
    for rtype, rels in _CLASS_SCHEMES[scheme].items():
        for rel in rels:
            SUBORD_COORD[rel] = SUBORD_COORD[rtype]

_flesh_out_subord('minsdrt')
# end SUBORD_COORD definition

# pylint: disable=no-init
class RfcConstraint(Enum):
    """
    What sort of right frontier constraint to apply during decoding:

        * simple: every relation is treated as subordinating
        * full: (falls back to simple in case of unlabelled prediction)
    """
    simple = 1
    full = 2
    none = 3

    @classmethod
    def from_string(cls, string):
        "command line arg to RfcConstraint"
        names = {x.name: x for x in cls}
        rfc = names.get(string)
        if rfc is not None:
            return rfc
        else:
            oops = "invalid choice: {}, choose from {}"
            choices = ", ".join('{}'.format(x) for x in names)
            raise ArgumentTypeError(oops.format(string, choices))
# pylint: enable=no-init


class DiscData(object):
    """
    Natural reading order decoding: incremental building of tree in order of
    text (one edu at a time)

    Basic discourse data for a state: chosen links between edus at that stage +
    right-frontier state. To save space, only new links are stored. the
    complete solution will be built with backpointers via the parent field

    RF: right frontier, = admissible attachment point of current discourse unit

    :param parent: parent state (previous decision)

    :param link: current decision (a triplet: target edu, source edu, relation)
    :type link: (string, string, string)

    :param tolink: remaining unattached discourse units
    :type tolink: [string]
    """
    def __init__(self, parent=None, accessible=None, tolink=None):
        self._accessible = accessible or []
        self.parent = parent
        self._link = None
        self._tolink = tolink or []

    def accessible(self):
        """return the list of edus that are on the right frontier

        :rtype: [string]
        """
        return self._accessible

    def final(self):
        "return `True` if there are no more links to be made"
        return self._tolink == []

    def tobedone(self):
        """return the list of edus to be linked

        :rtype: [string]
        """
        return self._tolink

    def last_link(self):
        "return the link that was made to get to this state, if any"
        return self._link

    def link(self, to_edu, from_edu, relation,
             rfc=RfcConstraint.full):
        """
        rfc = "full": use the distinction coord/subord
        rfc = "simple": consider everything as subord
        rfc = "none" no constraint on attachment
        """
        #if to_edu not in self.accessible():
        #    print("error: unreachable node", to_edu, "(ignored)", file= sys.stderr)
        if True:
            index = self.accessible().index(to_edu)
            self._link = (to_edu, from_edu, relation)
            # update the right frontier -- coord relations replace their
            # attachment points, subord are appended, and evrything below
            # disappear from the RF
            # unknown relations are subord
            #print(type(relation), file= sys.stderr)
            #print(map(type, SUBORD_COORD.values()), file= sys.stderr)
            if rfc == RfcConstraint.full and\
                SUBORD_COORD.get(relation, "subord") == "coord":
                self._accessible = self._accessible[:index]
            elif rfc == RfcConstraint.simple:
                self._accessible = self._accessible[:index + 1]
            elif rfc == RfcConstraint.none:
                pass
            else:
                raise Exception("Unknown RFC: {}".format(rfc))
            self._accessible.append(from_edu)

    def __str__(self):
        template = ("{link}/ "
                    "accessibility={accessibility}/ "
                    "to attach={to_attach}")
        return template.format(link=self._link,
                               accessibility=self._accessible,
                               to_attach=[str(x) for x in self._tolink])

    def __repr__(self):
        return str(self)



class DiscourseState(State):
    """
    Natural reading order decoding: incremental building of tree in order of
    text (one edu at a time)

    instance of discourse graph with probability for each attachement+relation on a subset
    of edges.

    implements the State interface to be used by Search

    strategy: at each step of exploration choose a relation between two edus
    related by probability distribution, reading order
    a.k.a NRO "natural reading order", cf Bramsen et al., 2006. in temporal processing.

    'data' is set of instantiated relations (typically nothing at the
    beginning, but could be started with a few chosen relations)

    'shared' points to shared data between states (here proba distribution
    between considered pairs of edus at least, but also can include precomputed
    info for heuristics)
    """
    def __init__(self, data, heuristics, shared):
        self._data = data  # needed for heuristics
        self._shared = shared
        super(DiscourseState, self).__init__(data,
                                             cost=0,
                                             future_cost=heuristics(self))


    def proba(self, edu_pair):
        """return the label and probability that an edu pair are attached,
        or `("no", None)` if we don't have a prediction for the pair

        :rtype: (string, float or None)"""
        return self._shared["probs"].get(edu_pair, ("no", None))

    def shared(self):
        "information shared between states"
        return self._shared

    def strategy(self):
        """ full or not, if the RFC is applied to labelled edu pairs
        """
        return self._shared["RFC"]

    # solution found when everything has been instantiated
    # TODO: adapt to disc parse, according to choice made for data
    def is_solution(self):
        return self.data().final()

    def _mk_score_transform(self):
        """return a function that converts scores depending on our
        configuration"""
        if self.shared()["use_prob"]:
            # pylint: disable=no-member
            return lambda x: -numpy.log(x)
            # pylint: enable=no-member
        else:
            return lambda x: x

    def next_states(self):
        """must return a state and a cost
        TODO: adapt to disc parse, according to choice made for data -> especially update to RFC
        """
        res = []
        one = self.data().tobedone()[0]
        transform = self._mk_score_transform()
        #print ">> taking care of node ", one
        for attachmt in self.data().accessible():
            new = copy.deepcopy(self.data())
            new.tobedone().pop(0)
            relation, prob = self.proba((attachmt, one))
            if prob is not None:
                new.link(attachmt, one, relation, rfc=self.strategy())
                new.parent = self.data()
                score = transform(prob)
                res.append((new, score))
        return res

    def __str__(self):
        return str(self.data()) + ": " + str(self.cost())


    def __repr__(self):
        return str(self.data()) + ": " + str(self.cost())


    # heuristiques
    # pylint: disable=no-self-use
    def h_zero(self):
        "always 0"
        return 0.
    # pylint: enable=no-self-use

    def h_average(self):
        """return the average probability possible when n nodes still need to be attached
        assuming the best overall prob in the distrib"""
        missing_links = self.data().tobedone()
        transform = self._mk_score_transform()
        return sum(transform(self.shared()["heuristics"]["average"][x])
                   for x in missing_links)

    def h_best_overall(self):
        """return the best probability possible when n nodes still need to be attached
        assuming the best overall prob in the distrib"""
        missing_links = len(self.data().tobedone())
        transform = self._mk_score_transform()
        prob = self.shared()["heuristics"]["best_overall"]
        return transform(prob) * missing_links

    def h_best(self):
        """return the best probability possible when n nodes still need to be attached
        assuming the best overall prob in the distrib"""
        missing_links = self.data().tobedone()
        transform = self._mk_score_transform()
        return sum(transform(self.shared()["heuristics"]["best_attach"][x])
                   for x in missing_links)

#########################################

class TwoStageNROData(DiscData):
    """similar as above with different handling of inter-sentence and intra-sentence relations

    accessible is list of starting edus (only one for now)
    """

    def __init__(self, parent=None, accessible=None, tolink=None):
        super(TwoStageNROData, self).__init__(parent=parent,
                                              accessible=accessible,
                                              tolink=tolink)
        self._accessible_global = accessible or []
        self._accessible_sentence = accessible or []
        self._intra = True
        self._current_sentence = 1

    def accessible(self):
        """
        wip:
        """
        if self._intra:
            return self._accessible_sentence
        else:
            return self._accessible_global

    def update_mode(self):
        "switch between intra/inter-sentential parsing mode"
        self._intra = not self._intra

    def link(self, to_edu, from_edu, relation):
        """WIP

        """
        index = self.accessible().index(to_edu)
        self._link = (to_edu, from_edu, relation)
        if self._intra: # simple RFC for now
            self._accessible_global = self._accessible_sentence[: index + 1]
            self._accessible_global.append(from_edu)
        else:
            self._accessible_sentence = self._accessible_sentence[:index + 1]
            self._accessible_global.append(from_edu)



class TwoStageNRO(DiscourseState):
    """similar as above with different handling of inter-sentence and intra-sentence relations"""
    def __init__(self):
        pass

    def same_sentence(self, edu1, edu2):
        """not implemented: will always return False
        TODO: this should go in preprocessing before launching astar
        ?? would it be easier to have access to all edu pair features ??
        (certainly for that one)
        """
        return self.shared().get("same_sentence", lambda x: False)(edu1, edu2)

    def next_states(self):
        """must return a state and a cost
        """
        # TODO: decode differently on intra/inter sentence
        res = []
        one = self.data().tobedone()[0]
        # inter/intra
        # if intra shared.sentence([one]) != current
        #      current += 1 / intra = False
        # else:
        #      intra = True


        for attachmt in self.data().accessible():
            new = copy.deepcopy(self.data())
            new.tobedone().pop(0)
            relation, prob = self.proba((attachmt, one))
            transform = self._mk_score_transform()
            if prob is not None:
                new.link(attachmt, one, relation)
                new.parent = self.data()
                res.append((new, transform(prob)))
        return res

###################################

class DiscourseSearch(Search):
    """
    subtype of astar search for discourse: should be the same for
    every astar decoder, provided the discourse state is a subclass
    of DiscourseState

    recover solution should be as is, provided a state has at least the following
    info:
    - parent: parent state
    - _link: the actual prediction made at this stage (1 state = 1 relation = (du1, du2, relation)
    """
    def new_state(self, data):
        return DiscourseState(data, self._h_func, self.shared())

    def recover_solution(self, endstate):
        "follow back pointers to collect list of chosen relations on edus."
        res = []
        current = endstate.data()
        while current.parent is not None:
            #print current
            res.append(current.last_link())
            current = current.parent
        res.reverse()
        return res



class DiscourseBeamSearch(DiscourseSearch, BeamSearch):
    pass


class Heuristic(Enum):
    """
    What sort of right frontier constraint to apply during decoding:

        * simple: every relation is treated as subordinating
        * full: (falls back to simple in case of unlabelled prediction)
    """
    zero = 0
    max = 1
    best = 2
    average = 3

    @classmethod
    def from_string(cls, string):
        "command line arg to Heuristic"
        names = {x.name: x for x in cls}
        val = names.get(string)
        if val is not None:
            return val
        else:
            oops = "invalid choice: {}, choose from {}"
            choices = ", ".join('{}'.format(x) for x in names)
            raise ArgumentTypeError(oops.format(string, choices))
# pylint: enable=no-init

HEURISTICS = {Heuristic.zero: DiscourseState.h_zero,
              Heuristic.max: DiscourseState.h_best_overall,
              Heuristic.best: DiscourseState.h_best,
              Heuristic.average: DiscourseState.h_average}


# pylint: disable=too-many-arguments
class AstarArgs(namedtuple('AstarArgs',
                           ['heuristics',
                            'rfc',
                            'beam',
                            'nbest',
                            'use_prob'])):
    """
    Configuration options for the A* decoder

    :param heuristics: an a* heuristic funtion (estimate the cost of what has
                       not been explored yet)
    :type heuristics: `Heuristic`

    :param use_prob: indicates if previous scores are probabilities in [0,1]
                     (to be mapped to -log) or arbitrary scores (untouched)
    :type use_prob: bool

    :param beam: size of the beam-search (if None: vanilla astar)
    :type beam: int or None

    :param rfc: what sort of right frontier constraint to apply
    :type rfc: RfcConstraint
    """
    def __new__(cls,
                heuristics=Heuristic.zero,
                beam=None,
                rfc=RfcConstraint.simple,
                use_prob=True,
                nbest=1):
        return super(AstarArgs, cls).__new__(cls,
                                             heuristics, rfc,
                                             beam, nbest, use_prob)
# pylint: enable=too-many-arguments



def preprocess_heuristics(prob_distrib):
    """precompute a set of useful information used by heuristics, such as
             - best probability
             - table of best probability when attaching a node, indexed on that node

    format of prob_distrib is format given in main decoder: a list of
    (arg1,arg2,proba,best_relation)
    """
    result = {}
    result["best_overall"] = max([x[2] for x in prob_distrib])
    result["best_attach"] = defaultdict(float)
    result["average"] = defaultdict(list)
    for du1, du2, prob, label in prob_distrib:
        result["best_attach"][du2.id] = max(result["best_attach"][du2.id], prob)
        result["average"][du2.id].append(prob)

    for one in result["average"]:
        result["average"][one] = sum(result["average"][one])/len(result["average"][one])
    #print(result, file= sys.stderr)
    return result


# TODO: order function should be a method parameter
# - root should be specified ? or a fake root ? for now, it is the first edu
# - should allow for (at least local) argument inversion (eg background), for more expressivity
# - dispatch of various strategies should happen here.
#   the original strategy should be called simpleNRO or NRO
class AstarDecoder(Decoder):
    """wrapper for astar decoder to be used by processing pipeline
    returns a structure, or nbest structures
    """
    def __init__(self, astar_args):
        self._heuristic = HEURISTICS[astar_args.heuristics]
        self._args = astar_args

    def decode(self, prob_distrib):
        probs = get_prob_map(prob_distrib)
        edus = [x.id for x in get_sorted_edus(prob_distrib)]
        print("\t %s nodes to attach"%(len(edus)-1), file=sys.stderr)

        search_shared = {"probs": probs,
                         "use_prob": self._args.use_prob,
                         "heuristics": preprocess_heuristics(prob_distrib),
                         "RFC": self._args.rfc}
        if self._args.beam:
            astar = DiscourseBeamSearch(heuristic=self._heuristic,
                                        shared=search_shared,
                                        queue_size=self._args.beam)
        else:
            astar = DiscourseSearch(heuristic=self._heuristic,
                                    shared=search_shared)
            genall = astar.launch(DiscData(accessible=[edus[0]], tolink=edus[1:]),
                                  norepeat=True, verbose=False)
        # nbest solutions handling
        all_solutions = []
        for _ in range(self._args.nbest):
            endstate = genall.next()
            sol = astar.recover_solution(endstate)
            all_solutions.append(sol)
            print("nbest=%d" % self._args.nbest, file=sys.stderr)
        return all_solutions
