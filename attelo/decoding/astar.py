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
from attelo.edu import EDU

_class_schemes = {
    "subord_coord": {
        "subord": set(["elaboration", "e-elab", "attribution", "comment", "flashback", "explanation", "alternation"]),
        "coord": set(["continuation", "parallel", "contrast", "temploc", "frame", "narration", "conditional", "result", "goal", "background"]),
        "NONE": set(["null", "unknown", "NONE"])
        },
    # four class +/- closer to PDTB
    "pdtb": {
        "contingency": set(["explanation", "conditional", "result", "goal", "background"]),
        "temporal": set(["temploc", "narration", "flashback"]),
        "comparison": set(["parallel", "contrast"]),
        "expansion": set(["frame", "elaboration", "e-elab", "attribution", "continuation", "comment", "alternation"]),
        "error": set(["null", "unknown", "NONE"])
        },
    # our own hierarchy
    "minsdrt": {
        "structural": set(["parallel", "contrast", "alternation", "conditional"]),
        "sequence": set(["result", "narration", "continuation"]),
        "expansion": set(["frame", "elaboration", "e-elab", "attribution", "comment", "explanation"]),
        "temporal": set(["temploc", "goal", "flashback", "background"]),
        "error": set(["null", "unknown", "NONE"])
        }
}


subord_coord = {"structural": "coord",
                "sequence": "coord",
                "expansion": "subord",
                "temporal": "subord",
                "error": "subord"}

for one in _class_schemes["minsdrt"]:
    for rel in _class_schemes["minsdrt"][one]:
        subord_coord[rel] = subord_coord[one]


# pylint: disable=no-init, too-few-public-methods
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
# pylint: enable=no-init, too-few-public-methods


class DiscData:
    """
    Natural reading order decoding: incremental building of tree in order of
    text (one edu at a time)

    basic discourse data for a state: chosen links between edus at that stage + right-frontier state
    to save space, only new links are stored. the complete solution will be
    built with backpointers via the parent field

    RF: right frontier, = admissible attachment point of current discourse unit
    parent: parent state (previous decision)
    link: current decision (a triplet: target edu, source edu, relation)
    tolink: remaining unattached discourse units
    """
    def __init__(self, parent=None, accessible=[], tolink=[]):
        self._accessible = accessible
        self.parent = parent
        self._link = None
        self._tolink = tolink

    def accessible(self):
        return self._accessible

    def final(self):
        return self._tolink == []

    def tobedone(self):
        return self._tolink

    def link(self, to_edu, from_edu, relation,
             RFC=RfcConstraint.full):
        """
        RFC = "full": use the distinction coord/subord
        RFC = "simple": consider everything as subord
        RFC = "none" no constraint on attachment
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
            #print(map(type, subord_coord.values()), file= sys.stderr)
            if RFC == RfcConstraint.full and\
                subord_coord.get(relation, "subord") == "coord":
                self._accessible = self._accessible[: index]
            elif RFC == RfcConstraint.simple:
                self._accessible = self._accessible[: index + 1]
            elif RFC == RfcConstraint.none:
                pass
            else:
                raise Exception("Unknown RFC: {}".format(RFC))
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
        super(DiscourseState, self).__init__(data,
                                             cost=0,
                                             future_cost=heuristics(self))
        self._shared = shared


    def data(self):
        return self._data

    def proba(self, edu_pair):
        return self._shared["probs"].get(edu_pair, ("no", None))

    def shared(self):
        return self._shared

    def strategy(self):
        """ full or not, if the RFC is applied to labelled edu pairs
        """
        return self._shared["RFC"]

    # solution found when everything has been instantiated
    # TODO: adapt to disc parse, according to choice made for data
    def is_solution(self):
        return self.data().final()


    def next_states(self):
        """must return a state and a cost
        TODO: adapt to disc parse, according to choice made for data -> especially update to RFC
        """
        all = []
        one = self.data().tobedone()[0]
        #print ">> taking care of node ", one
        for attachmt in self.data().accessible():
            # FIXME: this might is problematic because we use things like
            # checking if an EDU is in some list, which fails on object id
            new = copy.deepcopy(self.data())
            new.tobedone().pop(0)
            relation, pr = self.proba((attachmt, one))
            if pr is not None:
                new.link(attachmt, one, relation, RFC=self.strategy())
                new.parent = self.data()
                if self._shared["use_prob"]:
                    if pr == 0:
                        score = -numpy.inf
                    else:
                        score = -math.log(pr)
                    all.append((new, score))
                else:
                    all.append((new, pr))
        return all

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
        # return the average probability possible when n nodes still need to be attached
        # assuming the best overall prob in the distrib
        missing_links = self.data().tobedone()
        #try:
        if self.shared()["use_prob"]:
            transform = lambda x: -math.log(x) if x != 0 else -numpy.inf
        else:
            transform = lambda x: x
        try:
            pr = sum(transform(self.shared()["heuristics"]["average"][x])
                     for x in missing_links)
        except:
            print(missing_links, file= sys.stderr)
            print(self.shared()["heuristics"]["average"][x], file= sys.stderr)
            sys.exit(0)
        return pr

    def h_best_overall(self):
        # return the best probability possible when n nodes still need to be attached
        # assuming the best overall prob in the distrib
        missing_links = len(self.data().tobedone())
        pr = self.shared()["heuristics"]["best_overall"]
        if self.shared()["use_prob"]:
            score = -math.log(pr) if pr != 0 else -numpy.inf
            return score * missing_links
        else:
            return pr * missing_links


    def h_best(self):
        # return the best probability possible when n nodes still need to be attached
        # assuming the best overall prob in the distrib
        missing_links = self.data().tobedone()
        if self.shared()["use_prob"]:
            transform = lambda x: -math.log(x) if x != 0 else -numpy.inf
        else:
            transform = lambda x: x
        pr = sum(transform(self.shared()["heuristics"]["best_attach"][x])
                 for x in missing_links)
        return pr

#########################################

class TwoStageNROData(DiscData):
    """similar as above with different handling of inter-sentence and intra-sentence relations

    accessible is list of starting edus (only one for now)
    """

    def __init__(self, parent=None, accessible=[], tolink=[]):
        self._accessible_global = accessible
        self._accessible_sentence = accessible
        self.parent = parent
        self._link = None
        self._tolink = tolink
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
        self._intra = not(self._intra)

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
        all = []
        one = self.data().tobedone()[0]
        # inter/intra
        # if intra shared.sentence([one]) != current
        #      current += 1 / intra = False
        # else:
        #      intra = True


        for attachmt in self.data().accessible():
            # FIXME: this might is problematic because we use things like
            # checking if an EDU is in some list, which fails on object id
            new = copy.deepcopy(self.data())
            new.tobedone().pop(0)
            relation, pr = self.proba((attachmt, one))
            if pr is not None:
                new.link(attachmt, one, relation)
                new.parent = self.data()
                if self._shared["use_prob"]:
                    if pr == 0:
                        score = -numpy.inf
                    else:
                        score = -math.log(pr)
                    all.append((new, score))
                else:
                    all.append((new, pr))
        return all

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
        # follow back pointers to collect list of chosen relations on edus.
        all = []
        current = endstate.data()
        while current.parent is not None:
            #print current
            all.append(current._link)
            current = current.parent
        all.reverse()
        return all



class DiscourseBeamSearch(BeamSearch):

    def new_state(self, data):
        return DiscourseState(data, self._h_func, self.shared())

    def recover_solution(self, endstate):
        # follow back pointers to collect list of chosen relations on edus.
        all = []
        current = endstate.data()
        while current.parent is not None:
            #print current
            all.append(current._link)
            current = current.parent
        all.reverse()
        return all


# pylint: disable=too-few-public-methods
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
# pylint: enable=no-init, too-few-public-methods

HEURISTICS = {Heuristic.zero: DiscourseState.h_zero,
              Heuristic.max: DiscourseState.h_best_overall,
              Heuristic.best: DiscourseState.h_best,
              Heuristic.average: DiscourseState.h_average}


# pylint: disable=too-many-arguments, too-few-public-methods
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
    :type use_prob: Boolean

    :param beam: size of the beam-search (if None: vanilla astar)
    :type beam: Int or None

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
# pylint: enable=too-many-arguments, too-few-public-methods



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
    for a1, a2, p, r in prob_distrib:
        result["best_attach"][a2.id] = max(result["best_attach"][a2.id], p)
        result["average"][a2.id].append(p)

    for one in result["average"]:
        result["average"][one] = sum(result["average"][one])/len(result["average"][one])
    #print(result, file= sys.stderr)
    return result

def prob_distrib_convert(prob_distrib):
    """convert a probability distribution table to desired input for a* decoder
    NOT IMPLEMENTED: to be factored in from astar_decoder
    """
    pass




# TODO: order function should be a method parameter
# - root should be specified ? or a fake root ? for now, it is the first edu
# - should allow for (at least local) argument inversion (eg background), for more expressivity
# - dispatch of various strategies should happen here.
#   the original strategy should be called simpleNRO or NRO
def astar_decoder(prob_distrib,
                  astar_args,
                  **kwargs):
    """wrapper for astar decoder to be used by processing pipeline
    returns a structure, or nbest structures

    """
    prob = {}
    edus = set()
    for a1, a2, p, r in prob_distrib:
        #print r
        prob[(a1.id, a2.id)] = (r, p)
        edus.add((a1.id, int(a1.start)))
        edus.add((a2.id, int(a2.start)))

    edus = list(edus)
    edus.sort(key=lambda x: x[1])
    saved = edus
    edus = map(lambda x: x[0], edus)
    print("\t %s nodes to attach"%(len(edus)-1), file=sys.stderr)

    heuristic_function = HEURISTICS[astar_args.heuristics]
    search_shared = {"probs": prob,
                     "use_prob": astar_args.use_prob,
                     "heuristics": preprocess_heuristics(prob_distrib),
                     "RFC": astar_args.rfc}
    if astar_args.beam:
        a = DiscourseBeamSearch(heuristic=heuristic_function,
                                shared=search_shared,
                                queue_size=astar_args.beam)
    else:
        a = DiscourseSearch(heuristic=heuristic_function,
                            shared=search_shared)
    genall = a.launch(DiscData(accessible=[edus[0]], tolink=edus[1: ]),
                      norepeat=True, verbose=False)
    # nbest solutions handling
    all_solutions = []
    for i in range(astar_args.nbest):
        endstate = genall.next()
        sol = a.recover_solution(endstate)
        all_solutions.append(sol)
    print("nbest=%d" % astar_args.nbest, file=sys.stderr)
    if astar_args.nbest == 1:
        return sol
    else:
        return all_solutions
