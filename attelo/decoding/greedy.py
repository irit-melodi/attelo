'''
    July 2012

    @author: stergos

    implementation of the locally greedy approach similar with DuVerle & Predinger (2009, 2010)
    (but adapted for SDRT, where the notion of adjacency includes embedded segments)

'''
import sys
from pprint import pprint

from .util import get_sorted_edus


def areStrictlyAdjacent(one, two, edus):
    """ returns true in the following cases:

          [one] [two]
          [two] [one]

        in the rest of the cases (when there is an edu between one and two) it returns false
    """
    for edu in edus:
        if edu.id != one.id and edu.id != two.id:
            if one.end <= edu.start and edu.start <= two.start:
                return False
            if one.end <= edu.end and edu.end <= two.start:
                return False
            if two.end <= edu.start and edu.start <= one.start:
                return False
            if two.end <= edu.end and edu.end <= one.start:
                return False

    return True

def isEmbedded(one, two):
    """ returns true when one is embedded in two, that is:

            [two ... [one] ... ]

        returns false in all other cases
    """

    return two.id != one.id and two.start <= one.start and one.end <= two.end

def getNeigbours(instances):

    neighbours = dict()
    edus = get_sorted_edus(instances)

    for one in edus :
        theNeighbours = []
        theNeighbours_ids = set()
        for two in edus :
            if one.id != two.id:
                if areStrictlyAdjacent(one, two, edus):
                    if two.id not in theNeighbours_ids:
                        theNeighbours_ids.add(two.id)
                        theNeighbours.append(two)
                if isEmbedded(one, two) or isEmbedded(two, one):
                    if two.id not in theNeighbours_ids:
                        theNeighbours_ids.add(two.id)
                        theNeighbours.append(two)

        neighbours[one] = theNeighbours

    return neighbours

def locallyGreedy(instances, prob=True, use_prob=True):

    # get the probability distribution
    probabilityDistribution = dict()
    for s, t, p, r in instances :
        probabilityDistribution[(s.id, t.id)] = (p, r)

    neighbours = getNeigbours(instances)
    edus = get_sorted_edus(instances)

    attachments = []
    edus_id = set([x.id for x in edus])
    
    while len(edus) > 1:    
        print >> sys.stderr, len(edus), 
        highest = 0.0
        toRemove = None
        attachment = None
        newSpan = None

        for source in edus:
            for target in neighbours[source]:
                if (source.id, target.id) in probabilityDistribution:
                    if probabilityDistribution[(source.id, target.id)][0] > highest:
                        highest = probabilityDistribution[(source.id, target.id)][0]
                        toRemove = source
                        newSpan = target
                        attachment = (source.id, target.id, probabilityDistribution[(source.id, target.id)][1])

        if toRemove is not None :
            attachments.append(attachment)
            edus.remove(toRemove)
            edus_id = edus_id - set([toRemove.id])
            #print >> sys.stderr, edus_id
            # PM : added to propagate locality to percolated span heads
            neighbours[newSpan].extend(neighbours[toRemove])
            #print >> sys.stderr, neighbours[newSpan]
            neighbours[newSpan] = [x for x in neighbours[newSpan] if x.id in edus_id and not(x.id==newSpan.id)] 
        else: #stop if nothing to attach, but this is wrong
            #print >> sys.stderr, "warning: no attachment found"
            #print edus
            #print edus_id
            #print [neighbours[x] for x in edus]
            #sys.exit(0)
            edus = []
    print >> sys.stderr, ""
    return attachments

