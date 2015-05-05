"""
attelo subcommands
"""

# Author: Eric Kow
# License: CeCILL-B (French BSD3)

# pylint: disable=import-self
from . import\
    (enfold,
     inspect,
     graph,
     rewrite,
     report)

SUBCOMMANDS = [enfold,
               inspect,
               graph,
               rewrite,
               report]
