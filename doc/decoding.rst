Decoding
========

Joint decoding mode (AD.L and ADL)
----------------------------------
Joint decoding mode works with both the AD.L and the ADL schemes
(latter is not yet implemented at the time of this writing 2015-02-16).

In the AD.L scheme, we query the attachment model for an attachment
probability and the relation labelling model for its best labelling
probability. We then multiply these into a single probability score
for the decoder.

In the ADL scheme (ie. with only one model that does everything), we
merely retrieve the highest probability score for each given instance.

Note that joint decoding mode cannot be used with models that cannot
supply probabilities (for example, the perceptron). Post-label mode
must be used instead. (See :ref:`learning` for details)

Post-label decoding mode (AD.L and ADL)
---------------------------------------
In post-label mode we retrieve just the probability of attachment
(from the AD model in the AD.L case, and `1-P(UNRELATED)` in the
ADL case) and feed this to the decoder (along with a dummy
`UNKNOWN` label).

For each edge in the decoder output, we then retrieve the best label possible
from the labeling model (or the best non-UNRELATED label in the ADL case) and
apply that to the decoder outputs
