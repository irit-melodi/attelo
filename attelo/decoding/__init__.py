'''
Decoding in attelo consists in building discourse graphs from a
set of attachment/labelling predictions.
'''

# pylint: disable=too-few-public-methods

from __future__ import print_function

from .interface import (Decoder)
from .util import DecoderException

from .astar import (AstarDecoder)
from .baseline import LastBaseline, LocalBaseline
from .mst import (MsdagDecoder, MstDecoder, MstRootStrategy)
from .greedy import LocallyGreedy
from .local import (AsManyDecoder, BestIncomingDecoder)
