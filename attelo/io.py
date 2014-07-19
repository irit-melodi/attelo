"""
Saving and loading data or models
"""

from __future__ import print_function
import cPickle
import Orange
import sys
import traceback

# ---------------------------------------------------------------------
# feedback
# ---------------------------------------------------------------------


# pylint: disable=too-few-public-methods, redefined-builtin, invalid-name
class Torpor(object):
    """
    Announce that we're about to do something, then do it,
    then say we're done.

    Usage: ::

        with Torpor("doing a slow thing"):
            some_slow_thing

    Output (1): ::

        doing a slow thing...

    Output (2a): ::

        doing a slow thing... done

    Output (2b): ::

        doing a slow thing... ERROR
        <stack trace>

    :param quiet: True to skip the message altogether
    """
    def __init__(self, msg,
                 sameline=True,
                 quiet=False,
                 file=sys.stderr):
        self._msg = msg
        self._file = file
        self._sameline = sameline
        self._quiet = quiet

    def __enter__(self):
        if self._quiet:
            return
        elif self._sameline:
            print(self._msg, end="... ", file=self._file)
        else:
            print("[start]", self._msg, file=self._file)

    def __exit__(self, type, value, tb):
        if tb is None:
            if not self._quiet:
                done = "done" if self._sameline else "[end] " + self._msg
                print(done, file=self._file)
        else:
            if not self._quiet:
                oops = "ERROR!" if self._sameline else "ERROR! " + self._msg
                print(oops, file=self._file)
            traceback.print_exception(type, value, tb)
            sys.exit(1)
# pylint: enable=too-few-public-methods, redefined-builtin, invalid-name


# ---------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------


def read_data(attachments, relations, verbose=False):
    """
    Given an attachment file and a relations file (latter can
    be None, return their contents in table form)
    """
    with Torpor("Reading attachments", quiet=not verbose):
        data_attach = Orange.data.Table(attachments)

    if relations is None:
        data_relations = None
    else:
        with Torpor("Reading relations", quiet=not verbose):
            data_relations = Orange.data.Table(relations)
    return data_attach, data_relations


# ---------------------------------------------------------------------
# models
# ---------------------------------------------------------------------


# TODO: describe model type
def load_model(filename):
    """
    Load model into memory from file
    """
    with open(filename, "rb") as f:
        return cPickle.load(f)


def save_model(filename, model):
    """
    Dump model into a file
    """
    with open(filename, "wb") as f:
        cPickle.dump(model, f)
