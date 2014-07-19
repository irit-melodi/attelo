"""
Manipulating data tables (taking slices, etc)
"""


def related_attachments(phrasebook, table):
    """Return just the entries in the attachments table that
    represent related EDU pairs
    """
    return table.filter_ref({phrasebook.label: "True"})


def related_relations(phrasebook, table):
    """Return just the entries in the relations table that represent
    related EDU pair
    """
    return table.filter_ref({phrasebook.label: ["UNRELATED"]}, negate=1)


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
