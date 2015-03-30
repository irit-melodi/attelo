"""
Configuring the harness
"""

# Author: Eric Kow
# License: CeCILL-B (French BSD3-like)

from collections import namedtuple

from attelo.util import Team

# pylint: disable=too-few-public-methods


class Keyed(namedtuple('Keyed', 'key payload')):
    '''
    A keyed object is just any object that is attached with a
    short unique (mnemonic) identifier.

    Keys often appear in filenames so it's best to avoid
    whitespace, fancy characters, and for portability reasons,
    anything non-ASCII.
    '''
    pass


class LearnerConfig(Team):
    """
    Combination of an attachment and a relation learner variant

    :type attach: Learner
    :type relate: Learner
    """
    def __new__(cls, attach, relate):
        team = super(LearnerConfig, cls).__new__(cls, attach, relate)
        team.key = team.attach.key
        if team.relate is not None:
            team.key += "_" + team.relate.key
        return team


class EvaluationConfig(namedtuple("EvaluationConfig",
                                  "key settings learner decoder")):
    """
    Combination of learners, decoders and decoder settings
    for an attelo evaluation

    The settings can really be of type that has a 'key'
    field; but you should have a way of extracting at
    least a :py:class:`DecodingMode` from it

    :type learner: Keyed (Team learner)
    :type decoder: Keyed Decoder
    :type settings: Keyed (???)
    """
    @classmethod
    def simple_key(cls, learner, decoder):
        """
        generate a short unique name for a learner/decoder combo
        """
        return "%s-%s" % (learner.key, decoder.key)
