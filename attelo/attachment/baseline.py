"""
Baseline decoders
"""

def local_baseline(prob_distrib, threshold = 0.5, use_prob=True):
    """just attach locally if prob is > threshold
    """
    predicted = []
    for (arg1, arg2, probs, label) in prob_distrib:
        attach = probs
        if use_prob:
            if attach > threshold:
                predicted.append((arg1.id, arg2.id, label))
        else:
            if attach >= 0.0:
                predicted.append((arg1.id, arg2.id, label))
    return predicted


def last_baseline(prob_distrib, use_prob=True):
    "attach to last, always"
    edus = getSortedEDUs(prob_distrib)
    ordered_pairs = zip(edus[:-1],edus[1:])
    dict_prob = {}
    for (a1,a2,p,r) in prob_distrib:
        dict_prob[(a1.id,a2.id)]=(r,p)

    predicted=[(a1.id,a2.id,dict_prob[(a1.id,a2.id)][0]) for (a1,a2) in ordered_pairs]
    return predicted
