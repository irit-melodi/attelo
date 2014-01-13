class Features(object):
    """
    Mapping from examples to features.

    The parser works with almost an arbitrary set of features,
    but there is a minimal core that should be present in some
    form, basically

    * for a source and a target node:
        * its id
        * its text-span start
        * its text-span end
    * a label
    * some grouping of nodes into larger units (for example,
      the nodes for annotations in the same file)

    Dataset merely some indirection to deal with the fact that
    these features can have different names in different data,
    just use eg. `dataset.grouping` as a dictionary key
    """

    def __init__(self,
                 source="SOURCE",
                 target="TARGET",
                 target_span_start="TargetSpanStart",
                 target_span_end="TargetSpanEnd",
                 source_span_start="SourceSpanStart",
                 source_span_end="SourceSpanEnd",
                 grouping="FILE",
                 label="CLASS"):
        self.source=source
        self.target=target
        self.target_span_start=target_span_start
        self.target_span_end=target_span_end
        self.source_span_start=source_span_start
        self.source_span_end=source_span_end
        self.grouping=grouping
        self.label=label
