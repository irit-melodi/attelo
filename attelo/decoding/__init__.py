'''
Decoding in attelo consists in building discourse graphs from a
set of attachment/labelling predictions.

There are two kinds of modules in this package:

    * decoders: eg., baseline, greedy, mst: convert probability distributions
      into graphs

    * management: by rights the important bits of these should have been
      re-exported here so you'd never need to look into attelo.decoding.control
      or attelo.decoding.util unless you were writing a decoder yourself
'''

# pylint: disable=too-few-public-methods

from __future__ import print_function
from collections import namedtuple
import sys

from .interface import (Decoder)
from .util import DecoderException

from .astar import (AstarDecoder)
from .baseline import LastBaseline, LocalBaseline
from .mst import (MsdagDecoder, MstDecoder, MstRootStrategy)
from .greedy import LocallyGreedy
from .local import (AsManyDecoder, BestIncomingDecoder)


class DecoderArgs(namedtuple("DecoderAgs",
                             ["threshold",
                              "mst_root_strategy",
                              "astar",
                              "use_prob"])):
    """
    Superset of parameters needed by attelo decoders. Attelo decoders
    accept a wide variety of arguments, sometimes overlapping, often
    not. At the end of the day all these parameters find their way into
    data structure (sometimes hived off into sections like `astar`).
    We also provide below universal wrappers that pick out just the
    parameters needed by the individual decoders

    :param use_prob: `True` if model scores are probabilities in [0,1]
                     (to be mapped to -log), `False` if arbitrary scores
                     (to be untouched)
    :type use_prob: bool

    :param mst_root_strategy: How the MST/MSDAG decoders should select
                              their root node
    :type mst_root_strategy: :py:class:MstRootStrategy:

    :param threshold: For some decoders, a probability floor that helps
                      the decoder decide whether or not to attach something
    :type threshold: float or None

    :param astar: Config options specific to the A* decoder
    :type astar: AstarArgs
    """
    pass


def _mk_local_decoder(config, default=0.5):
    """
    Instantiate the local decoder
    """
    if config.threshold is None:
        threshold = default
        print("using default threshold of {}".format(threshold),
              file=sys.stderr)
    else:
        threshold = config.threshold
        print("using requested threshold of {}".format(threshold),
              file=sys.stderr)
    return LocalBaseline(threshold, config.use_prob)


def _mk_mst_decoder(config):
    """
    Instantiate an MST decoder
    """
    return MstDecoder(config.mst_root_strategy,
                      config.use_prob)


DECODERS = {"last": lambda _: LastBaseline(),
            "local": _mk_local_decoder,
            "locallyGreedy": lambda _: LocallyGreedy(),
            "msdag": lambda c: MsdagDecoder(c.mst_root_strategy, c.use_prob),
            "mst": _mk_mst_decoder,
            "astar": lambda c: AstarDecoder(c.astar),
            "asmany": lambda _: AsManyDecoder(),
            "bestin": lambda _: BestIncomingDecoder(),
           }
"""
Dictionary (`string -> DecoderAgs -> Decoder`) of decoder names (recognised by
the command line interface) to wrappers. Wrappers smooth out the differences
between decoders, making it so that each known decoder accepts the universal
:py:class:DecoderArgs:
"""
