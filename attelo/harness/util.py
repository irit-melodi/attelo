# Author: Eric Kow
# License: CeCILL-B (French BSD3-like)


"""
Miscellaneous utility functions
"""

from datetime import datetime, date
from os import path as fp
import hashlib
import os
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


def force_symlink(source, link_name, **kwargs):
    """
    Symlink from source to `link_name`, removing any
    prexisting file at `link_name`
    """
    if os.path.islink(link_name):
        os.unlink(link_name)
    elif os.path.exists(link_name):
        oops = "Can't force symlink from " + source +\
            " to " + link_name +\
            " because a file of that name already exists"
        raise Exception(oops)
    os.symlink(source, link_name, **kwargs)


def subdirs(parent):
    """
    Return all subdirectories within the parent dir
    (with combined path, ie. `parent/subdir`)
    """
    subpaths = (fp.join(parent, x) for x in os.listdir(parent))
    return filter(os.path.isdir, subpaths)


def makedirs(path, **kwargs):
    """
    Create a directory and its parents if it does not already
    exist
    """
    if not fp.exists(path):
        os.makedirs(path, **kwargs)


def md5sum_file(path, blocksize=65536):
    """
    Read a file and return its md5 sum
    """
    hasher = hashlib.md5()
    with open(path, 'rb') as afile:
        buf = afile.read(blocksize)
        while len(buf) > 0:
            hasher.update(buf)
            buf = afile.read(blocksize)
    return hasher.hexdigest()


def md5sum_dir(path, blocksize=65536):
    """
    Read a dir and return its md5 sum
    """
    hasher = hashlib.md5()
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            with open(filepath, 'rb') as afile:
                buf = afile.read(blocksize)
                while len(buf) > 0:
                    hasher.update(buf)
                    buf = afile.read(blocksize)
    return hasher.hexdigest()
