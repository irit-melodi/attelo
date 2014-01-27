class EDU(object):
    """ a class representing the EDU (id, span start and end, file) """

    def __init__(self, id, start, end, file):
        self._id = id
        self._start = start
        self._end = end
        self._file = file

    @property
    def id(self):
        return self._id

    @property
    def start(self):
        return self._start

    @property
    def end(self):
        return self._end

    @property
    def file(self):
        return self._file

    def __deepcopy__(self, memo):
        # edu.deepcopy here returns the EDU itself
        # this is (presumably) safe to do if we make all of the
        # members read-only
        return self

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
