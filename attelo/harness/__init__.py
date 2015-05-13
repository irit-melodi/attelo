"""
attelo experimental harness helpers

The modules here are meant to help with building your own
test harnesses around attelo. They provide opinionated
support for experiment layout and interfacing with attelo
"""

# Author: Eric Kow
# License: CeCILL-B (French BSD3)

from .config import (ClusterStage,
                     RuntimeConfig)
from .interface import (Harness)
