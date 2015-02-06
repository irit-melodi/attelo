.. _report:

Evaluation with attelo report
=============================

The ``attelo report`` command generates a set of evaluation reports
by comparing ``attelo decode`` results against a gold standard. So
far it creates:

* global precision/recall reports
* confusion matrices

At the moment, this command is assumed to be part of a larger
test harness setup in which ``attelo decode`` is run separately
over different combinations of fold, learner, decoder, etc;
with the output for each combination saved in a file. The test
harness in this setup must generate an index file that tells
attelo report what these combinatios are and where to find the
outputs that go with them.

Index file
----------

The index file is a json record with two parts

* folds: list of folds and an output directory associated
  with each fold
* configurations: list of configurations (learner/decoders combinations)
  and an output filename for that configuration. The idea here is that
  each fold directory contains the same set outputs (one for each
  configuration listed)

Folds
~~~~~

The fold list is a list of pairs, consisting of the fold number
and its path.  If the path is relative, it is interpreted as
being relative to the parent directory of the index file ::

   [{"number":0, "path":"fold-0"},
    {"number":1, "path":"fold-1"},
    {"number":2, "path":"fold-2"}]

Configuration
~~~~~~~~~~~~~
A configuration is combination of learners, decoders, and parameters
thereof.  You might have different combinations if for example you
wanted to compare how different decoders behaved on your data. The fields of
the configuration tuple are all strings:

* attach-learner: learning algorithm used for attachment
* relate-learner (optional): assumed to be the same as attach learner if
  not supplied
* decoder: decoder used
* predictions: filename for predictions made

Prediction filenames must be relative; they are interpreted as being
relative to the fold path (the idea here is that you would have the
same of outputs repeated across multiple folds) ::

    {"attach-learner":"bayes",
     "decoder":"locallyGreedy",
     "predictions":"output.bayes-locallyGreedy"},

The attach-learner, relate-learner, decoder fields are arbitrary strings
of your choosing. It is up to your test harness to associate these
strings with the different possible learner/decoder configurations.
For example, you might decide that you want to test two variants of the
MST decoder, one using the "fake root" strategy and one using the "leftmost
root" strategy.  In that case, your harness might choose to associate these
variants with the keys "mst-fake" and "mst-left".


Example index
~~~~~~~~~~~~~
::

    {"folds":[
        {"number":0, "path":"fold-0"},
        {"number":1, "path":"fold-1"},
        {"number":2, "path":"fold-2"},
        {"number":3, "path":"fold-3"},
        {"number":4, "path":"fold-4"},
        {"number":5, "path":"fold-5"},
        {"number":6, "path":"fold-6"},
        {"number":7, "path":"fold-7"},
        {"number":8, "path":"fold-8"},
        {"number":9, "path":"fold-9"}],
     "configurations":
         [{"attach-learner":"bayes",
           "decoder":"last",
           "predictions":"output.bayes-last"},
          {"attach-learner":"bayes",
           "decoder":"local",
           "predictions":"output.bayes-local"},
          {"attach-learner":"bayes",
           "decoder":"locallyGreedy",
           "predictions":"output.bayes-locallyGreedy"},
          {"attach-learner":"bayes",
           "decoder":"mst",
           "predictions":"output.bayes-mst"},
          {"attach-learner":"bayes",
           "decoder":"astar",
           "predictions":"output.bayes-astar"},
          {"attach-learner":"maxent",
           "decoder":"last",
           "predictions":"output.maxent-last"},
          {"attach-learner":"maxent",
           "decoder":"local",
           "predictions":"output.maxent-local"},
          {"attach-learner":"maxent",
           "decoder":"locallyGreedy",
           "predictions":"output.maxent-locallyGreedy"},
          {"attach-learner":"maxent",
           "decoder":"mst",
           "predictions":"output.maxent-mst"},
          {"attach-learner":"maxent",
           "decoder":"astar",
           "predictions":"output.maxent-astar"}]}
