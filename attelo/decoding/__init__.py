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

# pylint: disable=wildcard-import
# (just for re-export)
from .control import *
from .util import DecoderException
