For each grouping in your input data, ``attelo decode`` saves output
files in a number of different formats

CONLL style format
------------------

Our most recent format is an attempt at a CONLL-style output, with some
differences

-  we don't have a text field of any sort
-  because we are predicting graphs rather than depnedency trees we have
   a variable number of columns depending on the maximum number of
   parents a a given node can have

The number of columns is based on the maximum indegree of nodes in the
graph. At the very least, these columns will be present

-  id
-  grouping
-  start
-  end
-  parent1
-  label1

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

    ex51    group1      114.0   120.0   ex48    Elaboration    ex52    Background
    ex52    group1      121.0   129.0   ex48    Narration      ex51    Background
    ex55    group1      150.0   180.0   ex48    Parallel
    ex58    group1      199.0   204.0   ex48    Elaboration    ex55    Comment
    ex48    group1      11.0    102.0   0       ROOT
    ex61    group1      215.0   252.0   ex48    Explanation
    ex86    group2      563.0   568.0   0       ROOT
    ex68    group2      334.0   356.0   ex65    Alternation     ex71    Alternation
    ex85    group2      558.0   562.0   ex71    Background
    ex82    group2      531.0   538.0   ex71    QAP    ex79    Background

