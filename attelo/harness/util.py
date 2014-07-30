# Author: Eric Kow
# License: CeCILL-B (French BSD3-like)


"""
Miscellaneous utility functions
"""

from datetime import datetime, date
import subprocess
import sys


def timestamp():
    """
    Current date/time to minute resolution in an ISO format.
    """
    now = datetime.utcnow()
    return "%sT%s" % (date.isoformat(now.date()),
                      now.time().strftime("%H%M"))


def call(args, **kwargs):
    """
    Execute a command and die prettily if it fails
    """
    try:
        subprocess.check_call(args, **kwargs)
    except subprocess.CalledProcessError as err:
        sys.exit(err)
