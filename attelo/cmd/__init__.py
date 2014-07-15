"""
attelo subcommands
"""

# Author: Eric Kow
# License: CeCILL-B (French BSD3)

from . import\
    decode,\
    evaluate,\
    learn

SUBCOMMANDS = [learn, decode, evaluate]
