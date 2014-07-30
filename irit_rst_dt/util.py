# Author: Eric Kow
# License: CeCILL-B (French BSD3-like)


"""
Miscellaneous utility functions
"""

import os
from datetime import datetime, date
import subprocess
import sys

from .local import LOCAL_TMP


def timestamp():
    """
    Current date/time to minute resolution in an ISO format.
    """
    now = datetime.utcnow()
    return "%sT%s" % (date.isoformat(now.date()),
                      now.time().strftime("%H%M"))


def current_tmp():
    """
    Directory for the current run
    """
    return os.path.join(LOCAL_TMP, timestamp())


def latest_tmp():
    """
    Directory for last run (usually a symlink)
    """
    return os.path.join(LOCAL_TMP, "latest")


def call(args, **kwargs):
    """
    Execute a command and die prettily if it fails
    """
    try:
        subprocess.check_call(args, **kwargs)
    except subprocess.CalledProcessError as err:
        sys.exit(err)
