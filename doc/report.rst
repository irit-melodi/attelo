.. _report:

Understanding results
=====================

The ``attelo report`` synthesises individual counts into a single
report. As input, it requires a CSV formatted index file with the
following columns:

-  ``config`` - a string representing whatever combination of options
   you wanted to test (learner and decoder nome is a good example)
-  ``fold`` - the fold number (if you don't have folds, just use the
   same one everywhere)
-  ``counts_file`` - path to the individual count files for that
   config/fold pair

Typically you would use ``attelo report`` in your own evaluation
infrastructure. The idea would be to

1. generate folds ``attelo enfold``
2. learn/test on the folds ``attelo learn`` and ``attelo decode``,
   saving the counts
3. synthesise the results with ``attelo report``

