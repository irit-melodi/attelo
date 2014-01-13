class EDU(object):
    """ a class representing the EDU (id, span start and end, file) """

    def __init__(self, id, start, end, file):
        self.id = id
        self.start = start
        self.end = end
        self.file = file

    def __str__(self):
        return "EDU {}: ({}, {}) from {}".format(self.id, int(self.start), int(self.end), self.file)

    def __repr__(self):
        return "EDU {}: ({}, {}) from {}".format(self.id, int(self.start), int(self.end), self.file)

def mk_edu_pairs(features, domain):
    """
    Given a set of features a table domain, return a function that given an
    instance in the table, groups its features into a pair of edus.
    """
    arg1            = domain.index(features.source)
    arg2            = domain.index(features.target)
    targetSpanStart = domain.index(features.target_span_start)
    targetSpanEnd   = domain.index(features.target_span_end)
    sourceSpanStart = domain.index(features.source_span_start)
    sourceSpanEnd   = domain.index(features.source_span_end)
    FILE            = domain.index(features.grouping)

    def helper(x):
        return (EDU(x[arg1].value, x[sourceSpanStart].value, x[sourceSpanEnd].value, x[FILE].value),
                EDU(x[arg2].value, x[targetSpanStart].value, x[targetSpanEnd].value, x[FILE].value))
    return helper
