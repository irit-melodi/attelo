"""
Configuring the harness (see local.py)
"""

# Author: Eric Kow
# License: CeCILL-B (French BSD3-like)

from collections import namedtuple


# pylint: disable=pointless-string-statement, too-few-public-methods
class LearnerConfig(object):
    """
    Unique combination of learner settings.

    It is crucial that each configuration be associated with
    a unique name, as we use these names to determine if we can
    reuse a learner or not.
    """
    def __init__(self, name, attach, relate=None):
        self.name = name
        "unique name for this configuration"

        self.attach = attach
        "attachment learner"

        self.relate = relate
        "relation learner (None if same as attach)"

    @classmethod
    def simple(cls, name):
        """
        Simple learner consisting of just the attachment learner
        of the same name.
        """
        return cls(name, name)


_DecoderConfig = namedtuple("_DecoderConfig", "name decoder")


class DecoderConfig(_DecoderConfig):
    """
    Unique combination of decoder settings.
    """
    @classmethod
    def simple(cls, name):
        """
        Simple learner consisting of just the decoder of the same
        name.
        """
        return cls(name, name)
# pylint: enable=pointless-string-statement, too-few-public-methods

EvaluationConfig = namedtuple("EvaluationConfig", "name learner decoder")
