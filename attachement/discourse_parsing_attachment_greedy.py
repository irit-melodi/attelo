'''
    July 2012

    @author: stergos

    implementation of the locally greedy approach similar with DuVerle & Predinger (2009, 2010)
    (but adapted for SDRT, where the notion of adjacency includes embedded segments)

'''
import sys
from pprint import pprint

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

def getSortedEDUs(instances):
    edus_ids = set()
    edus = set()
    for (a1, a2, p, r) in instances:
        if a1.id not in edus_ids:
            edus_ids.add(a1.id)
            edus.add(a1)
        if a2.id not in edus_ids:
            edus_ids.add(a2.id)
            edus.add(a2)


    edus = list(edus)
    edus.sort(key = lambda x: int(x.start))

    return edus

def getNeigbours(instances):

    neighbours = dict()
    edus = getSortedEDUs(instances)

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
    edus = getSortedEDUs(instances)

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

