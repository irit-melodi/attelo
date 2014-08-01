"""
irit-rst-dt subcommands
"""

# Author: Eric Kow
# License: CeCILL-B (French BSD3)

from . import evaluate, features, gather, clean

SUBCOMMANDS = [gather, evaluate, features, clean]
