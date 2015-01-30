.. _output-format:

Output format
=============

The output format is the same as the EDU input format, except that
instead of a "potential parents" column, we have an number of columns
based on the maximum indegree of nodes in the graph.
At the very least, these columns will be present

- id
- text
- start
- end
- grouping
- parent1
- label1

If there are any nodes in the output graph with more than one parent
node, there will also be columns for them (padded if no link is present)

-  parent2
-  labelN
-  ..
-  parentN
-  labelN

If there is only one parent/label column, the result can be treated as
forming a dependency tree

::

    d1_492  anybody want sheep for wood?    dialogue_1  0   27  ROOT       ROOT    d1_493  Elaboration
    d1_493  nope, not me    dialogue_1  28  40  d1_492 Narration        d1_494  Parallel
    d1_494  not me either   dialogue_1  41  54  ROOT ROOT	d1_491	Alternation
