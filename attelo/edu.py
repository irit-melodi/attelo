"""
Uniquely identifying information for an EDU
"""

from collections import namedtuple

# pylint: disable=too-few-public-methods


class EDU(namedtuple("EDU", "id text start end grouping subgrouping")):
    """ a class representing the EDU
    (id, span start and end, grouping, subgrouping)
    """
    _str_template = ("EDU {id}: "
                     "({start}, {end}) "
                     "from {grouping} [{subgrouping}]"
                     "\t{text}")

    def __deepcopy__(self, _):
        # edu.deepcopy here returns the EDU itself
        # this is (presumably) safe to do if we make all of the
        # members read-only
        return self

    def __unicode__(self):
        return self._str_template.format(id=self.id,
                                         start=int(self.start),
                                         end=int(self.end),
                                         grouping=self.grouping,
                                         subgrouping=self.subgrouping,
                                         text=self.text)

    def __str__(self):
        return self._str_template.format(id=self.id,
                                         start=int(self.start),
                                         end=int(self.end),
                                         grouping=self.grouping,
                                         subgrouping=self.subgrouping,
                                         text=self.text)

    def span(self):
        """
        Starting and ending position of the EDU as an integer pair
        """
        return (int(self.start), int(self.end))


# pylint: disable=pointless-string-statement
FAKE_ROOT_ID = 'ROOT'
FAKE_ROOT = EDU(FAKE_ROOT_ID, '', 0, 0, None, None)
"""
a distinguished fake root EDU which simultaneously appears in
all groupings
"""
# pylint: enable=pointless-string-statement


# small helper for parsers
def edu_id2num(edu_id):
    """Get the number of an EDU"""
    edu_num = (int(edu_id.rsplit('_', 1)[1])
               if edu_id != FAKE_ROOT_ID
               else 0)
    return edu_num
