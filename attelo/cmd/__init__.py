"""
attelo subcommands
"""

# Author: Eric Kow
# License: CeCILL-B (French BSD3)

# pylint: disable=import-self
from . import\
    (decode,
     enfold,
     evaluate,
     inspect,
     graph,
     rewrite,
     learn,
     report)

SUBCOMMANDS = [learn,
               decode,
               enfold,
               evaluate,
               inspect,
               graph,
               rewrite,
               report]
