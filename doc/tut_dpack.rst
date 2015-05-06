
Datapacks and multipacks
========================

Attelo reads its `input files <../input>`__ into “datapacks”. Generally
speaking, we have one datapack per document, so when reading a corpus
in, we would be reading multiple datapacks (we read a multipack, ie. a
dictionary of datapacks, or perhaps a fancier structure in future attelo
versions)

.. code:: python

    from __future__ import print_function
    
    from os import path as fp
    from attelo.io import (load_multipack)
    
    CORPUS_DIR = 'example-corpus'
    PREFIX = fp.join(CORPUS_DIR, 'tiny')
    
    # load the data into a multipack
    mpack = load_multipack(PREFIX + '.edus',
                           PREFIX + '.pairings',
                           PREFIX + '.features.sparse',
                           PREFIX + '.features.sparse.vocab',
                           verbose=True)



.. parsed-literal::

    Reading edus and pairings... done [0 ms]
    Reading features... done [2 ms]
    Build data packs... done [0 ms]


As we can see below, multipacks are dictionaries from document names to
dpacks.

.. code:: python

    for dname, dpack in mpack.items():
        about = ("Doc: {name} |"
                 " edus: {edus}, pairs: {pairs},"
                 " features: {feats}")
        print(about.format(name=dname,
                           edus=len(dpack.edus),
                           pairs=len(dpack),
                           feats=dpack.data.shape))


.. parsed-literal::

    Doc: d2 | edus: 4, pairs: 9, features: (9, 7)
    Doc: d3 | edus: 3, pairs: 4, features: (4, 7)
    Doc: d1 | edus: 4, pairs: 9, features: (9, 7)


Datapacks store everything we know about a document:

-  edus: edus and their and their metadata
-  pairings: factors to learn on
-  data: feature array
-  target: predicted label for each instance

.. code:: python

    dpack = mpack.values()[0] # pick an arbitrary pack
    print("LABELS ({num}): {lbls}".format(num=len(dpack.labels), 
                                          lbls=", ".join(dpack.labels)))
    print()
    # note that attelo will by convention insert __UNK__ into the list of
    # labels, at position 0.  It also requires that UNRELATED and ROOT be
    # in the list of available labels
    
    for edu in dpack.edus[:3]:
        print(edu)
    print("...\n")
    
    for i, (edu1, edu2) in enumerate(dpack.pairings[:3]):
        lnum = dpack.target[i]
        lbl = dpack.get_label(lnum)
        feats = dpack.data[i,:].toarray()[0]
        print('PAIR', i, edu1.id, edu2.id, '\t|', lbl, '\t|', feats)
    print("...\n")
    
    for j, vocab in enumerate(dpack.vocab[:3]):
        print('FEATURE', j, vocab) 
    print("...\n")


.. parsed-literal::

    LABELS (6): __UNK__, elaboration, narration, continuation, UNRELATED, ROOT
    
    EDU ROOT: (0, 0) from None [None]	
    EDU d2_e2: (0, 27) from d2 [s3]	anybody want sheep for wood?
    EDU d2_e3: (28, 40) from d2 [s4]	nope, not me
    ...
    
    PAIR 0 ROOT d2_e2 	| elaboration 	| [ 0.  0.  0.  0.  0.  0.  0.]
    PAIR 1 d2_e3 d2_e2 	| narration 	| [ 1.  1.  0.  0.  0.  0.  0.]
    PAIR 2 d2_e4 d2_e2 	| UNRELATED 	| [ 2.  0.  1.  0.  0.  0.  0.]
    ...
    
    FEATURE 0 sentence_id_EDU2=1
    FEATURE 1 offset_diff_div3=0
    FEATURE 2 num_tokens_EDU2=19
    ...
    


There are a couple of datapack variants to be aware of:

-  *weighted* datapacks are parsed or partially parsed datapacks. They
   have a ``graph`` entry. We will explore weighted datapacks in the
   `parser tutorial <tut_parser>`__.
-  *stacked* datapacks: are formed by combining datapacks from different
   documents into one. Some parts of the attelo API (namely scoring and
   reporting) work with stacked datapacks. In the future (now:
   2015-05-06), they may evolve to deal with multipacks, in which case
   the notion of stack datapacks may dissapear

Conclusion
----------

This concludes our tour of attelo datapacks. In other tutorials we will
explore some of the uses of datapacks, namely as the input/output of our
`parsers <tut_parser>`__.
