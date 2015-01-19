"""
Uniquely identifying information for an EDU
"""

from collections import namedtuple

# pylint: disable=too-few-public-methods


class EDU(namedtuple("EDU_", "id text start end grouping")):
    """ a class representing the EDU (id, span start and end, grouping) """
    def __deepcopy__(self, _):
        # edu.deepcopy here returns the EDU itself
        # this is (presumably) safe to do if we make all of the
        # members read-only
        return self

    def __unicode__(self):
        return "EDU {}: ({}, {}) from {}\t{}".format(self.id,
                                                     int(self.start),
                                                     int(self.end),
                                                     self.grouping,
                                                     self.text)

    def __str__(self):
        return "EDU {}: ({}, {}) from {}\t{}".format(self.id,
                                                     int(self.start),
                                                     int(self.end),
                                                     self.grouping,
                                                     self.text)

    def __repr__(self):
        return str(self)

    def span(self):
        """
        Starting and ending position of the EDU as an integer pair
        """
        return (int(self.start), int(self.end))


# pylint: disable=pointless-string-statement
FAKE_ROOT_ID = '0'
FAKE_ROOT = EDU(FAKE_ROOT_ID, '', 0, 0, None)
"""
a distinguished fake root EDU which simultaneously appears in
all groupings
"""
# pylint: enable=pointless-string-statement
