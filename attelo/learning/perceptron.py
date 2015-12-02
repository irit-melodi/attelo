"""
A set of learner variants using a perceptron. The more advanced learners allow
for the possibility of structured prediction.

TODO:
- add more principled scores to probs conversion (right now, we do just 1-norm
  weight normalization and use expit function)
- add MC perc and PA for relation prediction.
- fold relation prediction into structured learning
"""

from __future__ import print_function
import sys
import time

from numpy.linalg import norm
from numpy import dot, zeros, sign
from scipy.special import expit  # aka the logistic function
import numpy as np

from attelo.decoding.util import (prediction_to_triples)
from attelo.metrics.tree import tree_loss
from attelo.table import (Graph, UNKNOWN)


# pylint: disable=invalid-name
# lots of mathy things here, so names may follow those conventions

class Perceptron(object):
    """Vanilla binary perceptron learner

    Parameters
    ----------
    n_iter: int, optional
        Number of passes over the training data (aka epochs).
        Defaults to 5.

    verbose: int, optional
        Verbosity level

    eta0: double, optional
        Constant by which the udpates are multiplied (aka constant
        learning rate).
        Defaults to 1.

    average: bool, optional
        When set to True, averages perceptrons weights from each
        iteration.

    use_prob: bool, optional
        When set to True, fake a notion of probabilities by using `log`
        tricks to return scores in [0, 1].
    """
    def __init__(self, n_iter=5, verbose=0, eta0=1.0,
                 # different from sklearn perceptron
                 average=False,
                 # should maybe not belong here
                 use_prob=False):
        self.nber_it = n_iter
        self.verbose = verbose
        self.eta0 = eta0
        self.avg = average
        self.use_prob = use_prob
        self.weights = None
        self.avg_weights = None
        self.can_predict_proba = use_prob

    def fit(self, X, Y):  # X contains all EDU pairs for corpus
        """ learn perceptron weights """
        self.init_model(X)
        self.learn(X, Y)
        return self

    def predict(self, X):
        W = self.avg_weights if self.avg else self.weights
        return sign(X.dot(W.T))

    def decision_function(self, X):
        W = self.avg_weights if self.avg else self.weights
        scores = X.dot(W.T)
        scores = scores.reshape(X.shape[0])  # lose 2nd dimension (== 1)
        return scores

    def init_model(self, X):
        verbose = self.verbose
        dim = X.shape[1]
        if verbose > 1:
            print("FEAT. SPACE SIZE:", dim)
        self.weights = zeros(dim, dtype='d')
        self.avg_weights = zeros(dim, dtype='d')

    def learn(self, X, Y):
        verbose = self.verbose
        if verbose > 1:
            print("-"*100, file=sys.stderr)
            print("Training...", file=sys.stderr)
            start_time = time.time()

        for n in xrange(self.nber_it):
            if verbose > 1:
                print("it. %3s \t" % n, file=sys.stderr)
                t0 = time.time()
            loss = 0.0
            inst_ct = 0
            for i in xrange(X.shape[0]):
                X_i = X[i]
                Y_i = Y[i]
                # track progress
                inst_ct += 1
                if verbose > 10:
                    sys.stderr.write("%s" % ("\b" * len(str(inst_ct)) +
                                             str(inst_ct)))
                # predict and update
                Y_hat, score = self._classify(X_i, self.weights)
                loss += self.update(Y_hat, Y_i, X_i, score)
            # progress in this iteration
            if inst_ct > 0:
                loss = loss / float(inst_ct)
            if verbose > 1:
                t1 = time.time()
                print("%s\tavg loss = %-7s" % (str(inst_ct),
                                               round(loss, 6)),
                      file=sys.stderr)
                print("\ttime = %-4s" % round(t1 - t0, 3), file=sys.stderr)
        if verbose > 1:
            elapsed_time = t1 - start_time
            print("done in %s sec." % round(elapsed_time, 3), file=sys.stderr)

    def update(self, Y_j_hat, Y_j, X_j, score):
        """ simple perceptron update rule"""
        rate = self.eta0
        X_j = X_j.toarray()
        error = (Y_j_hat != Y_j)
        W = self.weights
        if error:
            W = W + rate * Y_j * X_j
            self.weights = W
        if self.avg:
            self.avg_weights = self.avg_weights + W
        return int(error)

    def _classify(self, X, W):
        """ classify feature vector X using weight vector w into
        {-1,+1}"""
        score = float(X.dot(W.T))
        return sign(score), score


class PassiveAggressive(Perceptron):
    """Passive-Aggressive classifier in primal form.

    PA has a margin-based update rule: each update yields at least a
    margin of one (see details below). Specifically, we implement the
    PA-I rule for the binary setting (see Crammer et. al 2006).
    Setting parameter C=inf makes it equivalent to basic PA.

    Parameters
    ----------
    C: float
        Maximum step size (regularization). Also known as the
        aggressiveness parameter.
        `np.inf` makes it equivalent to basic PA.
        Defaults to 1.0.

    Notes
    -----
    TODO:
        [ ] implement the PA-II variant and add parameter "loss" to
        choose rule.
    """

    def __init__(self, C=1.0,
                 n_iter=5, verbose=0,
                 average=False,
                 use_prob=False):
        Perceptron.__init__(self,
                            n_iter=n_iter,
                            verbose=verbose,
                            eta0=1.0,
                            average=average,
                            use_prob=use_prob)
        self.C = C

    def update(self, Y_j_hat, Y_j, X_j, score):
        r"""PA-I update rule

        .. math::

           w = w + \tau y x \textrm{ where}

           \tau = min(C, \frac{loss}{||x||^2})

           loss  = \begin{cases}
                   0            & \textrm{if } margin \ge 1.0\\
                   1.0 - margin & \textrm{otherwise}
                   \end{cases}

           margin =  y (w \cdot x)
        """
        X_j = X_j.toarray()
        W = self.weights
        C = self.C
        margin = Y_j * score
        loss = 0.0
        if margin < 1.0:
            loss = 1.0 - margin
        norme = norm(X_j)
        if norme != 0:
            tau = loss / float(norme**2)
        tau = min(C, tau)
        W = W + tau * Y_j * X_j
        self.weights = W
        if self.avg:
            self.avg_weights = self.avg_weights + W
        return loss


class StructuredPerceptron(Perceptron):
    """ Perceptron classifier (in primal form) for structured
    problems."""

    def __init__(self, decoder,
                 n_iter=5, verbose=0, eta0=1.0,
                 average=False,
                 use_prob=False):
        Perceptron.__init__(self,
                            n_iter=n_iter,
                            verbose=verbose,
                            eta0=eta0,
                            average=average,
                            use_prob=use_prob)
        self.decoder = decoder

    def init_model(self, dim):
        verbose = self.verbose
        if verbose > 1:
            print("FEAT. SPACE SIZE:", dim)
        self.weights = zeros(dim, dtype='d')
        self.avg_weights = zeros(dim, dtype='d')

    def fit(self, datapacks, _targets):  # datapacks is an iterable
        """ learn struct. perceptron weights """
        self.init_model(datapacks[0].data.shape[1])
        self.learn(datapacks)
        return self

    def predict_score(self, dpack):
        return self.decision_function(dpack.data)

    def learn(self, datapacks):
        verbose = self.verbose
        if verbose > 1:
            print("-"*100, file=sys.stderr)
            print("Training struct. perc...", file=sys.stderr)
            start_time = time.time()

        for n in range(self.nber_it):
            if verbose > 1:
                print("it. %3s \t" % n, file=sys.stderr)
                t0 = time.time()
            loss = 0.0
            inst_ct = 0
            for dpack in datapacks:
                # extract data and target
                X = dpack.data  # each row is EDU pair
                Y = dpack.target  # each row is {-1,+1}
                # construct ref graph and mapping {edu_pair => index in X}
                ref_tree = []
                fv_index_map = {}
                for i, (edu1, edu2) in enumerate(dpack.pairings):
                    fv_index_map[edu1.id, edu2.id] = i
                    if Y[i] == 1:
                        ref_tree.append((edu1.id, edu2.id, UNKNOWN))
                # track progress
                inst_ct += len(ref_tree)  # was: 1
                if verbose > 10:
                    sys.stderr.write("%s" % ("\b" * len(str(inst_ct)) +
                                             str(inst_ct)))
                # predict tree based on current weight vector
                pred_tree = self._classify(dpack, X, self.weights)
                # tree loss
                tloss = self.update(pred_tree, ref_tree, X, fv_index_map)
                # from the tree loss, recover the absolute number of errors
                loss += int(tloss * len(ref_tree))
            # progress in this iteration
            avg_loss = loss / float(inst_ct)
            if verbose > 1:
                t1 = time.time()
                print("%s\tavg loss = %-7s" % (str(inst_ct),
                                               round(avg_loss, 6)),
                      file=sys.stderr)
                print("\ttime = %-4s" % round(t1-t0, 3), file=sys.stderr)
        if verbose > 1:
            elapsed_time = t1-start_time
            print("done in %s sec." % round(elapsed_time, 3), file=sys.stderr)

    def update(self, pred_tree, ref_tree, X, fv_map):
        rate = self.eta0
        # rt = [(t[0].span(),t[1].span()) for t in ref_tree]
        # pt = [(t[0].span(),t[1].span()) for t in pred_tree]
        # print("REF TREE:", rt)
        # print("PRED TREE:", pt)
        # print("INTER:", set(pt) & set(rt))
        W = self.weights
        # print("IN W:", W)
        # error = not( set(pred_tree) == set(ref_tree) )
        loss = tree_loss(ref_tree, pred_tree)
        if loss != 0:
            ref_fv = zeros(len(W), dtype='d')
            pred_fv = zeros(len(W), dtype='d')
            for ref_arc in ref_tree:
                id1, id2, _ = ref_arc
                ref_fv = ref_fv + X[fv_map[id1, id2]].toarray()
            for pred_arc in pred_tree:
                id1, id2, _ = pred_arc
                pred_fv = pred_fv + X[fv_map[id1, id2]].toarray()
            W = W + rate * (ref_fv - pred_fv)
        # print("OUT W:", W)
        self.weights = W
        if self.avg:
            self.avg_weights = self.avg_weights + W

        return loss

    def _classify(self, dpack, X, W):
        """ return predicted tree """
        num_items = len(dpack)
        # compute attachment scores for all EDU pairs
        scores = X.dot(W.T)  # TODO: should this be self.decision_function?
        scores = scores.reshape(num_items)  # lose 2nd dim (shape[1] == 1)
        # dummy labelling scores and predictions (for unlabelled parsing)
        unk = dpack.label_number(UNKNOWN)
        label = np.zeros((num_items, len(dpack.labels)))
        label[:, unk] = 1.0
        prediction = np.empty(num_items)
        prediction[:] = unk
        dpack = dpack.set_graph(Graph(prediction=prediction,
                                      attach=scores,
                                      label=label))
        # call decoder
        dpack_pred = self.decoder.transform(dpack)
        edge_list = prediction_to_triples(dpack_pred)
        return edge_list


class StructuredPassiveAggressive(StructuredPerceptron):
    """Structured PA-II classifier (in primal form) for structured
    problems.
    """

    def __init__(self, decoder,
                 C=1.0,
                 n_iter=5, verbose=0,
                 average=False,
                 use_prob=False):
        StructuredPerceptron.__init__(self, decoder,
                                      n_iter=n_iter,
                                      verbose=verbose,
                                      eta0=1.0,
                                      average=average,
                                      use_prob=use_prob)
        self.C = C

    def update(self, pred_tree, ref_tree, X, fv_map):
        r"""PA-II update rule:

        .. math::

            w = w + \tau * (\Phi(x,y)-\Phi(x-\hat{y})) \text{ where}

            \tau = min(C, \frac{loss}{||\Phi(x,y)-\Phi(x-\hat{y})||^2})

            loss = \begin{cases}
                   0             & \text{if } margin \ge 1.0\\
                   1.0 - margin  & \text{otherwise}
                   \end{cases}

            margin =  w \cdot (\Phi(x,y)-\Phi(x-\hat{y}))
        """
        W = self.weights
        C = self.C
        # compute Phi(x,y) and Phi(x,y^)
        ref_fv = zeros(len(W), dtype='d')
        pred_fv = zeros(len(W), dtype='d')
        for ref_arc in ref_tree:
            id1, id2, _ = ref_arc
            ref_fv = ref_fv + X[fv_map[id1, id2]].toarray()
        for pred_arc in pred_tree:
            id1, id2, _ = pred_arc
            pred_fv = pred_fv + X[fv_map[id1, id2]].toarray()
        # find tau
        delta_fv = ref_fv-pred_fv
        margin = float(dot(W, delta_fv.T))
        loss = 0.0
        tau = 0.0
        if margin < 1.0:
            loss = 1.0 - margin
        norme = norm(delta_fv)
        if norme != 0:
            tau = loss / float(norme**2)
        tau = min(C, tau)
        # update
        W = W + tau * delta_fv
        self.weights = W
        if self.avg:
            self.avg_weights = self.avg_weights + W
        return loss


def _score(w_vect, feat_vect, use_prob=False):
    score = dot(w_vect, feat_vect)
    if use_prob:
        score = expit(score)
    return score
# pylint: enable=no-member
