
Parsers
=======

An attelo parser converts “documents” (here: EDUs with some metadata)
into graphs (with EDUs as nodes and relation labels between them). In
API terms, a parser is something that enriches datapacks, progressively
adding or stripping away information until we get a full graph.

Parsers follow the scikit-learn estimator and transformer conventions,
ie. with a ``fit`` function to learn some model from training data and a
``transform`` function to convert (in our case) datapacks to enriched
datapacks.

Preliminaries
-------------

To begin our exploration of attelo parsers, let's load up a tiny
multipack of sample data.

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

    Reading edus and pairings... done [1 ms]
    Reading features... done [1 ms]
    Build data packs... done [0 ms]


We'll set aside one of the datapacks to test with, leaving the other two
for training. We do this by hand for this simple example, but you may
prefer to use the helper functions in
`attelo.fold <../api-doc/attelo#module-attelo.fold>`__ when working with
real data

.. code:: python

    test_dpack = mpack.values()[0]
    train_mpack = {k: mpack[k] for k in mpack.keys()[1:]}
    
    print('multipack entries:', len(mpack))
    print('train entries:', len(train_mpack))


.. parsed-literal::

    multipack entries: 3
    train entries: 2


Trying a parser out
-------------------

Now that we have our training and test data, we can try feeding them to
a simple parser. Before doing this, we'll take a quick detour to define
a helper function to visualise our parse results.

.. code:: python

    def print_results(dpack):
        'summarise parser results'
        for i, (edu1, edu2) in enumerate(dpack.pairings):
            wanted = dpack.get_label(dpack.target[i])
            got = dpack.get_label(dpack.graph.prediction[i])
            print(i, edu1.id, edu2.id, '\t|', got, '\twanted:', wanted)

As for parsing, we'll start with the attachment pipeline. It combines a
`learner <../api-doc/attelo.learning>`__ with a
`decoder <../api-doc/attelo.decoding>`__

.. code:: python

    from attelo.decoding.baseline import (LastBaseline)
    from attelo.learning import (SklearnAttachClassifier)
    from attelo.parser.attach import (AttachPipeline)
    from sklearn.linear_model import (LogisticRegression)
    
    learner = SklearnAttachClassifier(LogisticRegression())
    decoder = LastBaseline()
    parser1 = AttachPipeline(learner=learner, 
                             decoder=decoder)
    
    # train the parser
    train_dpacks = train_mpack.values()
    train_targets = [x.target for x in train_dpacks]
    parser1.fit(train_dpacks, train_targets)
            
    # now run on a test pack
    dpack = parser1.transform(test_dpack)
    print_results(dpack)


.. parsed-literal::

    0 ROOT d2_e2 	| __UNK__ 	wanted: elaboration
    1 d2_e3 d2_e2 	| UNRELATED 	wanted: narration
    2 d2_e4 d2_e2 	| UNRELATED 	wanted: UNRELATED
    3 ROOT d2_e3 	| UNRELATED 	wanted: continuation
    4 d2_e2 d2_e3 	| __UNK__ 	wanted: narration
    5 d2_e4 d2_e3 	| UNRELATED 	wanted: narration
    6 ROOT d2_e4 	| UNRELATED 	wanted: UNRELATED
    7 d2_e3 d2_e4 	| __UNK__ 	wanted: elaboration
    8 d2_e2 d2_e4 	| UNRELATED 	wanted: UNRELATED


Trying another parser
---------------------

In the output above, our predictions for every edge are either
``__UNK__`` or ``UNRELATED``. The attachment pipeline only predicts if
edges will be attached or not. What we need is to be able to predict
their labels.

.. code:: python

    from attelo.learning import (SklearnLabelClassifier)
    from attelo.parser.label import (SimpleLabeller)
    from sklearn.linear_model import (LogisticRegression)
    
    learner = SklearnLabelClassifier(LogisticRegression())
    parser2 = SimpleLabeller(learner=learner)
    
    # train the parser
    parser2.fit(train_dpacks, train_targets)
            
    # now run on a test pack
    dpack = parser2.transform(test_dpack)
    print_results(dpack)


.. parsed-literal::

    0 ROOT d2_e2 	| elaboration 	wanted: elaboration
    1 d2_e3 d2_e2 	| elaboration 	wanted: narration
    2 d2_e4 d2_e2 	| narration 	wanted: UNRELATED
    3 ROOT d2_e3 	| elaboration 	wanted: continuation
    4 d2_e2 d2_e3 	| elaboration 	wanted: narration
    5 d2_e4 d2_e3 	| narration 	wanted: narration
    6 ROOT d2_e4 	| elaboration 	wanted: UNRELATED
    7 d2_e3 d2_e4 	| elaboration 	wanted: elaboration
    8 d2_e2 d2_e4 	| narration 	wanted: UNRELATED


That doesn't quite look right. Now we have labels, but none of our edges
are ``UNRELATED``. But this is because the simple labeller will apply
labels on all unknown edges. What we need is to be able to combine the
attach and label parsers in a parsing pipeline

Parsing pipeline
----------------

A parsing pipeline is a parser that combines other parsers in sequence.
For purposes of learning/fitting, the individual steps can be thought of
as being run in parallel (in practice, they are fitted in sequnce). For
transforming though, they are run in order. A pipeline thus refines a
datapack over the course of multiple parsers.

.. code:: python

    from attelo.parser.pipeline import (Pipeline)
    
    # this is actually attelo.parser.full.PostlabelPipeline
    parser3 = Pipeline(steps=[('attach', parser1),
                              ('label', parser2)])
    
    parser3.fit(train_dpacks, train_targets)
    dpack = parser3.transform(test_dpack)
    print_results(dpack)


.. parsed-literal::

    0 ROOT d2_e2 	| elaboration 	wanted: elaboration
    1 d2_e3 d2_e2 	| UNRELATED 	wanted: narration
    2 d2_e4 d2_e2 	| UNRELATED 	wanted: UNRELATED
    3 ROOT d2_e3 	| UNRELATED 	wanted: continuation
    4 d2_e2 d2_e3 	| elaboration 	wanted: narration
    5 d2_e4 d2_e3 	| UNRELATED 	wanted: narration
    6 ROOT d2_e4 	| UNRELATED 	wanted: UNRELATED
    7 d2_e3 d2_e4 	| elaboration 	wanted: elaboration
    8 d2_e2 d2_e4 	| UNRELATED 	wanted: UNRELATED


Conclusion (for now)
--------------------

We have now seen some basic attelo parsers, how they use the
scikit-learn fit/transform idiom, and we can combine them with
pipelines. In future tutorials we'll break some of the parsers down into
their constituent parts (notice the attach pipeline is itself a
pipeline) and explore the process of writing parsers of our own.

