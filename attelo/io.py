"""
Saving and loading data or models
"""

from __future__ import print_function
import cPickle
import Orange
import sys
import time
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
        self._start = 0
        self._end = 0

    def __enter__(self):
        # we grab the wall time instead of using time.clock() (A)
        # because we # are not using this for profiling but just to
        # get a rough idea what's going on, and (B) because we want
        # to include things like IO into the mix
        self._start = time.time()
        if self._quiet:
            return
        elif self._sameline:
            print(self._msg, end="... ", file=self._file)
        else:
            print("[start]", self._msg, file=self._file)

    def __exit__(self, type, value, tb):
        self._end = time.time()
        if tb is None:
            if not self._quiet:
                done = "done" if self._sameline else "[-end-] " + self._msg
                ms_elapsed = 1000 * (self._end - self._start)
                final_msg = u"{} [{:.0f} ms]".format(done, ms_elapsed)
                print(final_msg, file=self._file)
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


def load_model(filename):
    """
    Load model into memory from file

    :rtype: Orange.classification.Classifier
    """
    with open(filename, "rb") as stream:
        return cPickle.load(stream)


def save_model(filename, model):
    """
    Dump model into a file

    :type: model: Orange.classification.Classifier
    """
    with open(filename, "wb") as stream:
        cPickle.dump(model, stream)
