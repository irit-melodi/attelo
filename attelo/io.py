"""
Saving and loading data or models
"""

import cPickle
import Orange

# ---------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------


def read_data(attachments, relations):
    """
    Given an attachment file and a relations file (latter can
    be None, return their contents in table form)
    """
    data_attach = Orange.data.Table(attachments)
    data_relations = Orange.data.Table(relations) if relations else None
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
