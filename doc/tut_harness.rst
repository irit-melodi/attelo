Harnesses
=========

In the previous tutorials, we introduced the notion of parsers, broke them down
into their constituent parts, and very briefly touched upon the idea of mixing
and matching parsers to form more interesting combinations.

If you find yourself in a situation where you have several parsing ideas that
you would like to explore, you may find it helpful to create an experimental
harness. A harness can be useful for

1. [reliability, convenience] bundling all the evaluation steps into a single
   easy-to-remember command (this eliminates the risk of omitting a crucial
   step)

2. [convenience] consistently generating an detailed report including
   confusion matrices, discriminating features, some visual samples of
   the output

3. [performance] caching shareable results to save evaluation time (both
   horizontally, for example, across parsers that can share models, and
   vertically, perhaps across different versions of a decoder but using the
   same model)

4. [performance] managing concurrency and distributed evaluation, which may
   be attractive if you have access to a compute cluster

The `attelo.harness` provides a basic framework for defining such
harnesses.  You would need to implement the `Harness` class, specifying

* the data to read
* a list of parsers to run (wrapped in `attelo.harness.config.EvaluationConfig`)
* some functions for assigning filenames to intermediary results
* and a variety of reporting options (for example, which evaluations you
  would like to generate extra reports on)

Have a look at the `example harness
<https://github.com/irit-melodi/attelo/blob/master/attelo/harness/example.py>`_
to get started, and perhaps also the `irit-rst-dt
<https://github.com/irit-melodi/irit-rst-dt>`_ to see how this might be
used in a real experimental setting.

Caching
-------
Attelo's caching mechanism uses the `cache` keyword argument in
`attelo.parser.Parser.fit` (cache is an attelo-ism, and is not standard to the
scikit estimator/transformer idiom). The idea is for parsers to accept a
dictionary from simple cache keywords (eg. 'attach') to paths. Parsers could
interact with the cache in different ways. In the simplest case, they might
look for a particular keyword to determine if there is a cache entry that
it could load (or should save to). Alternatively, if multiple parsers are
composed of parsers that they have in commone, they can avoid repeating work on
their constituent parts by simply passing their cache dictionaries down
(NB: it is up to parser authors to ensure that cache keys do not conflict;
parsers should document their cache keys in the API)

The `attelo.harness.Harness.model_paths` function implemented by your harness
should return exactly such a dictionary, as we might see in the example below

.. code:: python

    def model_paths(self, rconf, fold):
        if fold is None:
            parent_dir = self.combined_dir_path()
        else:
            parent_dir = self.fold_dir_path(fold)

        def _eval_model_path(mtype):
            "Model for a given loop/eval config and fold"
            bname = self._model_basename(rconf, mtype, 'model')
            return fp.join(parent_dir, bname)

        return {'attach': _eval_model_path("attach"),
                'label': _eval_model_path("label")}

Cluster mode: parallel and distributed
--------------------------------------
The attelo harness provides some crude support on a cluster:

* decoding is split into one decoding job per document/grouping; as each
  parser is learned [fit] (sequentially), the harness adds its decoding jobs
  [transform] to a pool of jobs in progress.
* each fold is self-contained, and can be run concurrently. If you are on
  a cluster with multiple machines reading from a shared filesystem, you
  can farm the folds out to separate machines (nb: the harness itself does
  not do this for you, so you would need to write eg. a shell script that
  does this parceling out of folds, but it can be broken down in a way that
  facilitates this usage, ie. with “initialise”, “run folds 1 and 2”,
  “run folds 3 and 4”, … “gather the results” as discrete steps)
