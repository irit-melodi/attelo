"""
Manipulating data tables (taking slices, etc)
"""

# pylint: disable=pointless-string-statement
UNRELATED = "UNRELATED"
"distinguished value for unrelateted relation labels"
# pylint: enable=pointless-string-statement


def AtteloTableException(Exception):
    def __init__(self, msg):
        super(AtteloTableException, self).__init__(msg)


def _related_condition(table, key, value):
    """
    Return a condition for `table.filter` that picks out instances
    for which the value associated with `key` is `value`.

    This would be just `{key: value}` except for the edge case where
    `value` is not in the table
    """
    if value in table.domain[key].values:
        cond_values = [value]
    else:
        cond_values = []
    return {key: cond_values}


def related_attachments(phrasebook, table):
    """Return just the entries in the attachments table that
    represent related EDU pairs
    """
    cond = _related_condition(table, phrasebook.label, 'True')
    return table.filter_ref(cond)


def related_relations(phrasebook, table):
    """Return just the entries in the relations table that represent
    related EDU pair
    """
    cond = _related_condition(table, phrasebook.label, UNRELATED)
    return table.filter_ref(cond, negate=1)


def _subtable_in_grouping(phrasebook, grouping, table):
    """Return the entries in the table that belong in the given
    group
    """
    return table.filter_ref({phrasebook.grouping: grouping})


def select_data_in_grouping(phrasebook, grouping, data_attach, data_relations):
    """Return only the data that belong in the given group
    """
    attach_instances = _subtable_in_grouping(phrasebook,
                                             grouping,
                                             data_attach)
    if data_relations:
        rel_instances = _subtable_in_grouping(phrasebook,
                                              grouping,
                                              data_relations)
    else:
        rel_instances = None
    return attach_instances, rel_instances


def select_edu_pair(phrasebook, pair, data):
    """Select the entry from a table corresponding to a single EDU pair

    Return None if the pair is not in the table.
    Raise an exception if there is more than one entry.
    """
    arg1, arg2 = pair
    selected = data.filter_ref({phrasebook.source: [arg1],
                                phrasebook.target: [arg2]})
    size = len(selected)
    if size == 0:
        return None
    elif size == 1:
        return selected[0]
    else:
        oops = "Found more than one entry for pair %s" % pair
        raise AtteloTableException(oops)


def index_by_metas(instances, metas=None):
    """transform a data table to a dictionary of instances indexed by ordered
    tuple of all meta-attributes; convenient to find instances associated to
    multiple tables (eg edu pairs for attachment+relations)
    """
    if metas is None:
        to_keep = lambda x: x.get_metas().values()
    else:
        to_keep = lambda x: [x[y] for y in metas]
    result = [(tuple([y.value for y in to_keep(x)]), x) for x in instances]
    return dict(result)
