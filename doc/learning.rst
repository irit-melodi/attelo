Learning
========
For each EDU pair, attelo computes a combined score by multiplying

* an attachment score (often a probability)
* the highest labelling probability

In the general case both attachment and labelling scores are probabilities,
and so the resulting score is also a probability; however, this is not always
appropriate for all classifiers.

For example, see this blog post on the implications of using a `hinge loss
function <http://mark.reid.name/blog/proper-losses-inevitability-of-rediscovery.html>`_
as opposed to the proper loss. If you are using a non-probability-based learner,
you should also set `--non-prob-scores` to false on decoding time
