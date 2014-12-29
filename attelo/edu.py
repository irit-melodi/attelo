"""
Uniquely identifying information for an EDU
"""

from collections import namedtuple


# pylint: disable=too-few-public-methods
class EDU(namedtuple("EDU_", "id start end file")):
    """ a class representing the EDU (id, span start and end, file) """

    def __deepcopy__(self, _):
        # edu.deepcopy here returns the EDU itself
        # this is (presumably) safe to do if we make all of the
        # members read-only
        return self

    def __str__(self):
        return "EDU {}: ({}, {}) from {}".format(self.id,
                                                 int(self.start),
                                                 int(self.end),
                                                 self.file)

    def __repr__(self):
        return str(self)

    def span(self):
        """
        Starting and ending position of the EDU as an integer pair
        """
        return (int(self.start), int(self.end))
# pylint: enable=too-few-public-methods


def mk_edu_pairs(phrasebook, domain):
    """
    Given a feature phrasebook a table domain, return a function that
    given an instance in the table, groups its features into a
    pair of edus.
    """
    arg1 = domain.index(phrasebook.source)
    arg2 = domain.index(phrasebook.target)
    tgt_start = domain.index(phrasebook.target_span_start)
    tgt_end = domain.index(phrasebook.target_span_end)
    src_start = domain.index(phrasebook.source_span_start)
    src_end = domain.index(phrasebook.source_span_end)
    grouping = domain.index(phrasebook.grouping)

    def mk_pairs(inst):
        "instance -> (EDU, EDU)"
        return (EDU(inst[arg1].value,
                    inst[src_start].value,
                    inst[src_end].value,
                    inst[grouping].value),
                EDU(inst[arg2].value,
                    inst[tgt_start].value,
                    inst[tgt_end].value,
                    inst[grouping].value))
    return mk_pairs
