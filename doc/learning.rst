Learning
========
In what follows,

* A refers to the attachment task: given an edu pair, is there a link between
  the two edus in any direction?
* D refers to the direction task: given an edu pair with a link between them,
  is the link from the textually earlier edu to the later on or vice-versa?
* L refers to the labelling task: given a directed linked edu pair, what is the
  label on edges between them?
* We use a '.' character to denote the grouping of the tasks into models, so
  for example, an 'AD.L' scheme is one in which we use one model for predicting
  attachement and directions together; and a separate model for predicting
  labels

AD.L scheme
-----------
In the AD.L scheme, we learn

* a binary attachment/direction model on all edu pairs — (e1, e2) and (e2, e1)
  are considered to be different pairs here
* a multiclass label model on only the edu pairs that are have an edge between
  them in the training data

See :ref:`decoding` for details on how these models are used on decoding time

Probabilities
-------------
In the general case both attachment and labelling scores are probabilities,
and so the resulting score is also a probability; however, this is not always
appropriate for all classifiers.

For example, see this blog post on the implications of using a `hinge loss
function <http://mark.reid.name/blog/proper-losses-inevitability-of-rediscovery.html>`_
as opposed to the proper loss.  If you are using a non-probability-based
learner,
you should also set `--non-prob-scores` to false on decoding time

Developers' note: if you are developing classifiers for attelo, and your
classifier does not return probabilties, it should implement
`decision_function` instead
