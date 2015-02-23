.. _input-format:

Input format
============

Input to attelo consists of three files two of which are aligned:

* an EDU input file with one line per discourse unit
* a pairings file with one line per EDU *pair*
* a features file also with one line per EDU *pair*

EDU inputs
----------

* global id: used by your application, arbitrary string?
  (NB: `ROOT` is a special name: no EDU should be named that,
  but all EDUs can have ROOT as a potential parent)
* text: essentially for debugging purposes, used by attelo
  graph to provide a visualisation of parses
* grouping (eg. file name, dialogue id): edus are only ever
  connected with edus in the same group. Also, folds are
  built on the basis of EDU groupings
* subgrouping (eg. sentence id): any common subunit that
  can hold multiple EDUs (use the EDU id itself if there
  is no useful notion of subgrouping).  Some decoders may
  try to treat links between EDUs in the same subgrouping
  differently from the general case
* span start: (int): used by decoders to order EDUs and
  determine their adjacency
* span end: (int): see span start

::

    d1_492	sheep for wood?	dialogue_1	sent1	0	15
    d1_493	nope, not me	dialogue_1	sent2	16	28
    d1_494	not me either	dialogue_1	sent2	29	42

Pairings
--------
The pairings file is a tab-delimited list of (parent, child) pairs,
with each element being either an EDU global id (from the EDU inputs),
or the distinguished label ROOT.  Each row in this file is corresponds with a
row in the feature files ::


    ROOT	d1_492
    d1_493	d1_492
    d1_494	d1_492
    ROOT	d1_493
    d1_492	d1_493
    d1_494	d1_493
    ROOT	d1_494
    d1_492	d1_494
    d1_493	d1_494


Note that attelo can also accept pairings files with a third column (which
it ignores)

Features
--------

Features and labels are supplied as in (multiclass) libsvm/svmlight format.

Relation labels
~~~~~~~~~~~~~~~
You should supply a single comment at the very beginning of the file,
which attelo can use to associate relation labels with string values ::

    # labels: <space delimited list of labels>

The labels 'UNRELATED' must exist and be used for any edu pairs which are not
related/attached.  For example, in the below, the second and fourth EDU pairs
are not considered to be related ::

    # labels: elaboration narration continuation UNRELATED ROOT
    1 1:1 2:1
    4 1:2
    2 1:3 3:1
    4 1:1
    3 1:2

Also, if intersentential learning/decoding is used, the label 'ROOT' must also
be exist and be used for links from the ROOT edu.


Note that labels are assumed to start from 1.

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
