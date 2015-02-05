.. _output-format:

Output format
=============

The output format is similar to the EDU pairings format. It is a tab-delimited
text file divided into rows and columns.  The columns are

* parent EDU id
* child EDU id
* relation label (or UNRELATED if no link between the two)

::

        ROOT	d1_492	ROOT
        d1_493	d1_492	UNRELATED
        d1_494	d1_492	UNRELATED
        ROOT	d1_493	UNRELATED
        d1_492	d1_493	elaboration
        d1_494	d1_493	result
        ROOT	d1_494	UNRELATED
        d1_492	d1_494	narration
        d1_493	d1_494	UNRELATED


The output above corresponds to the graph below ::


                ROOT
                  |
                  | ROOT
                  V
                d1_492 ------------------+
                  |                      |
                  | elaboration          | narration
                  V                      V
                d1_493 <---[result]-- d1_494


You can visualise the results with the `attelo report` (see :doc:`report`) and
attelo graph commands
