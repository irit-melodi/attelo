.. _input-format:

Input format
============

Input to attelo consists of two aligned files:

* an EDU input file with one line per discourse unit
* a features file with one line per EDU *pair*

The pairs are defined via a list of "potential parent edus" that
each edu can be associated with.

Suppose we have three EDUs (e1, e2, e3), and they respectively
list their potential parents as [e2, ROOT], [e1, e3], and [ROOT, e1].
In this case, the first two rows of the features file would
correspond to the e1 pairings, followed by another two rows for
the e2 pairings, and a final two rows for the e3 pairings:

1. e2,   e1
2. ROOT, e1
3. e1,   e2
4. e3,   e2
5. ROOT, e3
6. 1,    e3

EDU inputs
----------

* global id: used by your application, arbitrary string?
  (NB: `ROOT` is a special name: no EDU should be named that,
  but all EDUs can have ROOT as a potential parent)
* text: good for debugging
* grouping: eg. file, dialogue
* span start: (int)
* span end: (int)
* possible parents (single column, space delimited,
  NB: ROOT as distinguished name for root)

::

    d1_492  anybody want sheep for wood?    dialogue_1  0   27  ROOT d1_493 d1_494
    d1_493  nope, not me    dialogue_1  28  40  ROOT d1_492 d1_494
    d1_494  not me either   dialogue_1  41  54  ROOT d1_491 d1_492 d1_493

Features
--------

Features and labels are supplied as in (multiclass) libsvm/svmlight format.

Relation labels
~~~~~~~~~~~~~~~
You should supply a single comment at the very beginning of the file,
which attelo can use to associate relation labels with string values ::

    # labels: <space delimited list of labels>

As an example of what a features file might look like ::

    # labels: elaboration narration continuation
    1 1:1 2:1
    0 1:2
    2 1:3 3:1
    0 1:1
    3 1:2

If you do not supply this information, attelo will make do without it;
but some decoders may suffer as they use the relation labels in their
decision making process.

Categorical features
~~~~~~~~~~~~~~~~~~~~
Attelo no longer provides direct support for categorical features, that is,
features whose possible values are members of a set (eg. POS tag).  You should
perform `one hot encoding
<http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html>`_
on any categorical features you have. Luckily, with the svmlight sparse format,
this can be done with no additional cost in space and also opens the door for
more straightforward filtering on your part.

Other notes on features
~~~~~~~~~~~~~~~~~~~~~~~
Don't forget that the order that features appear in must correspond to the
order that pairings appear in the EDU file
