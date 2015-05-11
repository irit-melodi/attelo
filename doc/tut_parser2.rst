
Parsers (part 2)
================

In the previous tutorial, we saw a couple of basic parsers, and also
introduced the notion of a pipeline parser. It turns out that some of
the parsers we introduced and had taken for granted are themselves
pipelines. In this tutorial we will break these pipelines down and
explore some of finer grained tasks that a parser can do.

Preliminaries
-------------

We begin with the same multipacks and the same breakdown into a training
and test set

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
    
    test_dpack = mpack.values()[0]
    train_mpack = {k: mpack[k] for k in mpack.keys()[1:]}
    train_dpacks = train_mpack.values()
    train_targets = [x.target for x in train_dpacks]
    
    def print_results(dpack):
        'summarise parser results'
        for i, (edu1, edu2) in enumerate(dpack.pairings):
            wanted = dpack.get_label(dpack.target[i])
            got = dpack.get_label(dpack.graph.prediction[i])
            print(i, edu1.id, edu2.id, '\t|', got, '\twanted:', wanted)


.. parsed-literal::

    Reading edus and pairings... done [1 ms]
    Reading features... done [1 ms]
    Build data packs... done [0 ms]


Breaking a parser down (attach)
-------------------------------

If we examine the `source code for the attach
pipeline <https://github.com/irit-melodi/attelo/blob/master/attelo/parser/attach.py>`__,
we can see that it is in fact a two step pipeline combining the attach
classifier wrapper and a decoder. So let's see what happens when we run
the attach classifier by itself.

.. code:: python

    import numpy as np
    from attelo.learning import (SklearnAttachClassifier)
    from attelo.parser.attach import (AttachClassifierWrapper)
    from sklearn.linear_model import (LogisticRegression)
    
    def print_results_verbose(dpack):
        """Print detailed parse results"""
        for i, (edu1, edu2) in enumerate(dpack.pairings):
            attach = "{:.2f}".format(dpack.graph.attach[i])
            label = np.around(dpack.graph.label[i,:], decimals=2)
            got = dpack.get_label(dpack.graph.prediction[i])
            print(i, edu1.id, edu2.id, '\t|', attach, label, got)
            
    learner = SklearnAttachClassifier(LogisticRegression())
    parser1a = AttachClassifierWrapper(learner)
    parser1a.fit(train_dpacks, train_targets)
    
    dpack = parser1a.transform(test_dpack)
    print_results_verbose(dpack)


.. parsed-literal::

    0 ROOT d2_e2 	| 0.44 [ 1.  1.  1.  1.  1.  1.] __UNK__
    1 d2_e3 d2_e2 	| 0.43 [ 1.  1.  1.  1.  1.  1.] __UNK__
    2 d2_e4 d2_e2 	| 0.43 [ 1.  1.  1.  1.  1.  1.] __UNK__
    3 ROOT d2_e3 	| 0.44 [ 1.  1.  1.  1.  1.  1.] __UNK__
    4 d2_e2 d2_e3 	| 0.97 [ 1.  1.  1.  1.  1.  1.] __UNK__
    5 d2_e4 d2_e3 	| 0.39 [ 1.  1.  1.  1.  1.  1.] __UNK__
    6 ROOT d2_e4 	| 0.01 [ 1.  1.  1.  1.  1.  1.] __UNK__
    7 d2_e3 d2_e4 	| 0.42 [ 1.  1.  1.  1.  1.  1.] __UNK__
    8 d2_e2 d2_e4 	| 0.39 [ 1.  1.  1.  1.  1.  1.] __UNK__


Parsers and weighted datapacks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the output above, we have dug a little bit deeper into our datapacks.
Recall above that a parser translates datapacks to datapacks. The output
of a parser is always a *weighted datapack*., ie. a datapack whose
'graph' attribute is set to a record containing

-  attachment weights
-  label weights
-  predictions (like target values)

So called "standalone" parsers will take an unweighted datapack
(``graph == None``) and produce a weighted datapack with predictions
set. But some parsers tend to be more useful as part of a pipeline:

-  the attach classfier wrapper fills the attachment weights
-  likewise the label classifier wrapper assigns label weights
-  a decoder assigns predictions from weights

We see the first case in the above output. Notice that the attachments
have been set to values from a model, but the label weights and
predictions are assigned default values.

NB: all parsers should do "something sensible" in the face of all
inputs. This typically consists of assuming the default weight of 1.0
for unweighted datapacks.

Decoders
~~~~~~~~

Having now transformed a datapack with the attach classifier wrapper,
let's now pass its results to a decoder. In fact, let's try a couple of
different decoders and compare the output.

.. code:: python

    from attelo.decoding.baseline import (LocalBaseline)
    
    decoder = LocalBaseline(threshold=0.4)
    dpack2 = decoder.transform(dpack)
    print_results_verbose(dpack2)


.. parsed-literal::

    0 ROOT d2_e2 	| 0.44 [ 1.  1.  1.  1.  1.  1.] __UNK__
    1 d2_e3 d2_e2 	| 0.43 [ 1.  1.  1.  1.  1.  1.] __UNK__
    2 d2_e4 d2_e2 	| 0.43 [ 1.  1.  1.  1.  1.  1.] __UNK__
    3 ROOT d2_e3 	| 0.44 [ 1.  1.  1.  1.  1.  1.] __UNK__
    4 d2_e2 d2_e3 	| 0.97 [ 1.  1.  1.  1.  1.  1.] __UNK__
    5 d2_e4 d2_e3 	| 0.39 [ 1.  1.  1.  1.  1.  1.] UNRELATED
    6 ROOT d2_e4 	| 0.01 [ 1.  1.  1.  1.  1.  1.] UNRELATED
    7 d2_e3 d2_e4 	| 0.42 [ 1.  1.  1.  1.  1.  1.] __UNK__
    8 d2_e2 d2_e4 	| 0.39 [ 1.  1.  1.  1.  1.  1.] UNRELATED


The result above is what we get if we run a decoder on the output of the
attach classifier wrapper. This is in fact, the the same thing as
running the attachment pipeline. We can define a similar pipeline below.

.. code:: python

    from attelo.parser.pipeline import (Pipeline)
    
    # this is basically attelo.parser.attach.AttachPipeline
    parser1 = Pipeline(steps=[('attach weights', parser1a),
                              ('decoder', decoder)])
    parser1.fit(train_dpacks, train_targets)
    print_results_verbose(parser1.transform(test_dpack))


.. parsed-literal::

    0 ROOT d2_e2 	| 0.44 [ 1.  1.  1.  1.  1.  1.] __UNK__
    1 d2_e3 d2_e2 	| 0.43 [ 1.  1.  1.  1.  1.  1.] UNRELATED
    2 d2_e4 d2_e2 	| 0.43 [ 1.  1.  1.  1.  1.  1.] UNRELATED
    3 ROOT d2_e3 	| 0.44 [ 1.  1.  1.  1.  1.  1.] UNRELATED
    4 d2_e2 d2_e3 	| 0.97 [ 1.  1.  1.  1.  1.  1.] __UNK__
    5 d2_e4 d2_e3 	| 0.39 [ 1.  1.  1.  1.  1.  1.] UNRELATED
    6 ROOT d2_e4 	| 0.01 [ 1.  1.  1.  1.  1.  1.] UNRELATED
    7 d2_e3 d2_e4 	| 0.42 [ 1.  1.  1.  1.  1.  1.] __UNK__
    8 d2_e2 d2_e4 	| 0.39 [ 1.  1.  1.  1.  1.  1.] UNRELATED


Mixing and matching
-------------------

Being able to break parsing down to this level of granularity lets us
experiment with parsing techniques by composing different parsing
substeps in different ways. For example, below, we write two slightly
different pipelines, one which sets labels separately from decoding, and
one which combines attach and label scores before handing them off to a
decoder.

.. code:: python

    from attelo.learning.local import (SklearnLabelClassifier)
    from attelo.parser.label import (LabelClassifierWrapper, 
                                     SimpleLabeller)
    from attelo.parser.full import (AttachTimesBestLabel)
    
    learner_l = SklearnLabelClassifier(LogisticRegression())
    
    print("Post-labelling")
    print("--------------")
    parser = Pipeline(steps=[('attach weights', parser1a),
                             ('decoder', decoder),
                             ('labels', SimpleLabeller(learner_l))])
    parser.fit(train_dpacks, train_targets)
    print_results_verbose(parser.transform(test_dpack))
    
    print()
    print("Joint")
    print("-----")
    parser = Pipeline(steps=[('attach weights', parser1a),
                             ('label weights', LabelClassifierWrapper(learner_l)),
                             ('attach times label', AttachTimesBestLabel()),
                             ('decoder', decoder)])
    parser.fit(train_dpacks, train_targets)
    print_results_verbose(parser.transform(test_dpack))


.. parsed-literal::

    Post-labelling
    --------------
    0 ROOT d2_e2 	| 0.44 [ 0.    0.45  0.28  0.28  0.    0.  ] elaboration
    1 d2_e3 d2_e2 	| 0.43 [ 0.    0.4   0.34  0.25  0.    0.  ] elaboration
    2 d2_e4 d2_e2 	| 0.43 [ 0.    0.3   0.53  0.17  0.    0.  ] narration
    3 ROOT d2_e3 	| 0.44 [ 0.    0.45  0.28  0.28  0.    0.  ] elaboration
    4 d2_e2 d2_e3 	| 0.97 [ 0.    0.52  0.03  0.45  0.    0.  ] elaboration
    5 d2_e4 d2_e3 	| 0.39 [ 0.    0.37  0.43  0.2   0.    0.  ] UNRELATED
    6 ROOT d2_e4 	| 0.01 [ 0.    0.45  0.28  0.28  0.    0.  ] UNRELATED
    7 d2_e3 d2_e4 	| 0.42 [ 0.    0.41  0.35  0.24  0.    0.  ] elaboration
    8 d2_e2 d2_e4 	| 0.39 [ 0.    0.37  0.43  0.2   0.    0.  ] UNRELATED
    
    Joint
    -----
    0 ROOT d2_e2 	| 0.19 [ 0.    0.45  0.28  0.28  0.    0.  ] UNRELATED
    1 d2_e3 d2_e2 	| 0.17 [ 0.    0.4   0.34  0.25  0.    0.  ] UNRELATED
    2 d2_e4 d2_e2 	| 0.23 [ 0.    0.3   0.53  0.17  0.    0.  ] UNRELATED
    3 ROOT d2_e3 	| 0.19 [ 0.    0.45  0.28  0.28  0.    0.  ] UNRELATED
    4 d2_e2 d2_e3 	| 0.50 [ 0.    0.52  0.03  0.45  0.    0.  ] elaboration
    5 d2_e4 d2_e3 	| 0.17 [ 0.    0.37  0.43  0.2   0.    0.  ] UNRELATED
    6 ROOT d2_e4 	| 0.00 [ 0.    0.45  0.28  0.28  0.    0.  ] UNRELATED
    7 d2_e3 d2_e4 	| 0.17 [ 0.    0.41  0.35  0.24  0.    0.  ] UNRELATED
    8 d2_e2 d2_e4 	| 0.17 [ 0.    0.37  0.43  0.2   0.    0.  ] UNRELATED


Conclusion
----------

Thinking of parsers as transformers from weighted datapacks to weighted
datapacks should allow for some interesting parsing experiments, parsers
that

-  divide the work using different strategies on different subtypes of
   input (eg. intra vs intersentential links), or
-  work in multiple stages, maybe modifying past decisions along the
   way, or
-  influence future parsing stages by tweaking the weights they might
   see, or
-  prune out undesirable edges (by setting their weights to zero), or
-  apply some global constraint satisfaction algorithm across the
   possible weights

With a notion of a parsing pipeline, you should also be able to build
parsers that combine different experiments that you want to try
simultaneously
