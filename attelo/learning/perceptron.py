"""
A set of perceptron-like learners.

The more advanced learners allow for the possibility of structured
prediction.

TODO
----
[ ] add more principled scores to probs conversion (right now, we do
    just 1-norm weight normalization and use expit function)
[ ] add multiclass perc and PA for relation prediction
[ ] fold relation prediction into structured learning
"""

from __future__ import print_function
from math import sqrt
import sys
import time

from numpy.linalg import norm
from numpy import dot, zeros, sign
from scipy.special import expit  # aka the logistic function
import numpy as np

from attelo.decoding.util import prediction_to_triples
from attelo.metrics.tree import tree_loss
from attelo.metrics.util import oracle_ctree_spans
from attelo.table import Graph, UNKNOWN


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
        upd = self.eta0
        X_j = X_j.toarray()
        error = (Y_j_hat != Y_j)
        W = self.weights
        if error:
            W = W + upd * Y_j * X_j
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
    C : float
        Maximum step size (regularization). Also known as the
        aggressiveness parameter.
        `np.inf` makes it equivalent to basic PA.
        Defaults to 1.0.

    loss : string, optional
        The loss function to be used:
        hinge: PA-I variant from paper
        squared_hinge: PA-II variant from paper
    """

    def __init__(self, C=1.0,
                 n_iter=5, verbose=0,
                 loss="hinge",
                 average=False,
                 use_prob=False):
        Perceptron.__init__(self,
                            n_iter=n_iter,
                            verbose=verbose,
                            eta0=1.0,
                            average=average,
                            use_prob=use_prob)
        self.C = C
        self.loss = loss

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
        # learning rate, should be defined in fit(), passed to parent fit()
        lr = "pa1" if self.loss == "hinge" else "pa2"
        # end should be in fit()

        X_j = X_j.toarray()
        W = self.weights
        C = self.C
        # rename to match sklearn naming
        p = score

        # loss.loss(p, y)
        z = p * Y_j
        if z <= 1.0:  # Hinge, threshold=1.0
            loss_py = 1.0 - z  # threshold - z
        else:
            loss_py = 0.0
        # end loss.loss(p, y)

        if lr == "pa1":
            x_norm = norm(X_j)
            if x_norm == 0:
                upd = 0
            else:
                upd = x_norm**2
                upd = min(C, loss_py / upd)
        else:  # "pa2"
            x_norm = norm(X_j)
            upd = x_norm**2
            upd = loss_py / (upd + 0.5 / C)

        # sign the update
        upd *= Y_j

        # update weights
        W = W + upd * X_j
        self.weights = W

        # update the average weights
        if self.avg:
            self.avg_weights = self.avg_weights + W

        return loss_py


# =================================================
# structured prediction
# =================================================

# cost functions: tree loss functions

def dtree_loss(tree_gold, tree_pred, att_edus):
    """Dependency tree loss (mere wrapper).

    Parameters
    ----------
    tree_gold: list of (string, string, string)
        List of gold edges

    tree_pred: list of (string, string, string)
        List of predicted edges

    att_edus: list of EDUs
        List of attelo EDUs on which tree_gold and tree_pred are defined

    Returns
    -------
    Fraction of incorrectly predicted edges.
    """
    # discard EDUs, they are useless for this cost fun
    return tree_loss(tree_gold, tree_pred)


def ctree_loss(tree_gold, tree_pred, att_edus):
    """Constituency tree loss.

    Parameters
    ----------
    tree_gold: list of (string, string, string)
        List of gold edges

    tree_pred: list of (string, string, string)
        List of predicted edges

    att_edus: list of EDUs
        List of attelo EDUs on which tree_gold and tree_pred are defined

    Returns
    -------
    Fraction of incorrectly predicted spans.
    """
    spans_gold = oracle_ctree_spans(tree_gold, att_edus)
    spans_pred = oracle_ctree_spans(tree_pred, att_edus)
    if len(spans_gold) == 0:
        # TODO? incorporate this special case to
        # attelo.metrics.tree.tree_loss()
        # when given spans from a constituency tree,
        # this can be normal, but this should never happen for
        # edges from a dependency tree
        tloss = 0.0
    else:
        # call the standard tree_loss function
        tloss = tree_loss(spans_gold, spans_pred)
    return tloss


class StructuredPerceptron(Perceptron):
    """ Perceptron classifier (in primal form) for structured
    problems.

    Parameters
    ----------
    cost : string, optional
        The cost function to be used:
        dtree: dependency tree loss
        ctree: constituency tree loss (TODO)
    """

    # TODO refactor cost functions as classes like the loss functions
    # used in sklearn.linear_model
    cost_functions = {
        "ctree": ctree_loss,
        "dtree": dtree_loss,
    }

    def __init__(self, decoder,
                 n_iter=5, verbose=0, eta0=1.0,
                 cost="dtree",
                 average=False,
                 use_prob=False):
        Perceptron.__init__(self,
                            n_iter=n_iter,
                            verbose=verbose,
                            eta0=eta0,
                            average=average,
                            use_prob=use_prob)
        self.decoder = decoder
        self.cost = cost
        # validate params
        if self.cost not in self.cost_functions:
            raise ValueError("cost {} is not supported".format(self.cost))
        self.cost_function = self._get_cost_function(self.cost)

    def _get_cost_function(self, cost):
        """Get concrete cost function for str ``cost``."""
        try:
            cost_fun = self.cost_functions[cost]
        except KeyError:
            raise ValueError("Unsupported cost function {}".format(cost))
        return cost_fun

    def init_model(self, dim):
        verbose = self.verbose
        if verbose > 1:
            print("FEAT. SPACE SIZE:", dim)
        self.weights = zeros(dim, dtype='d')
        self.avg_weights = zeros(dim, dtype='d')

    def fit(self, datapacks, _targets, nonfixed_pairs=None):
        """Learn structured perceptron weights.

        Parameters
        ----------
        datapacks : iterable of DataPack
            TODO
        _targets : iterable of iterable of integers
            Label for each pairing in each datapack.
        nonfixed_pairs : list of list of integers
            List of indices of the nonfixed pairs, that should be considered
            when fitting a classifier.
        """
        self.init_model(datapacks[0].data.shape[1])
        self.learn(datapacks, nonfixed_pairs=nonfixed_pairs)
        return self

    def predict_score(self, dpack, nonfixed_pairs=None):
        """Get prediction scores.

        Currently returns attachment scores.

        Parameters
        ----------
        dpack: TODO
            TODO
        nonfixed_pairs: TODO
            TODO

        Returns
        -------
        scores: array of floats
            Predicted attachment score for each pairing of the DataPack ;
            if nonfixed_pairs is not None, the scores of fixed pairings
            are preserved.
        """
        num_items = len(dpack)
        if nonfixed_pairs is None:
            nonfixed_pairs = np.arange(num_items)
        if dpack.graph is None:
            scores = np.zeros(num_items)
        else:
            scores = np.copy(dpack.graph.attach)
        # compute decision scores for nonfixed pairs, using the model
        scores[nonfixed_pairs] = self.decision_function(
            dpack.data[nonfixed_pairs])
        return scores

    def learn(self, datapacks, nonfixed_pairs=None):
        if nonfixed_pairs is None:
            nonfixed_pairs = [None for dpack in datapacks]

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
            for dpack_idx, dpack in enumerate(datapacks):
                # nonfixed pairs for this dpack ; can be None
                nf_pairs = nonfixed_pairs[dpack_idx]
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
                inst_ct += 1
                if verbose > 10:
                    sys.stderr.write("%s" % ("\b" * len(str(inst_ct)) +
                                             str(inst_ct)))
                # predict tree based on current weight vector
                pred_tree = self._classify(dpack, X, self.weights,
                                           nonfixed_pairs=nf_pairs)
                # structured, cost sensitive loss
                loss_py = self.update(pred_tree, ref_tree, X, fv_index_map,
                                      dpack.edus,
                                      nonfixed_pairs=nf_pairs)
                # from the tree loss, recover the absolute number of errors
                loss += loss_py
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

    def update(self, pred_tree, ref_tree, X, fv_map, edus,
               nonfixed_pairs=None):
        """
        Parameters
        ----------
        edus : list(attelo.edu.EDU)
            List of attelo EDUs on which pred_tree and ref_tree are
            defined.
        """
        upd = self.eta0
        # rt = [(t[0].span(),t[1].span()) for t in ref_tree]
        # pt = [(t[0].span(),t[1].span()) for t in pred_tree]
        # print("REF TREE:", rt)
        # print("PRED TREE:", pt)
        # print("INTER:", set(pt) & set(rt))
        W = self.weights
        # print("IN W:", W)
        # error = not( set(pred_tree) == set(ref_tree) )

        # compute Phi(x,y) and Phi(x,y_hat)
        ref_fv = zeros(len(W), dtype='d')
        pred_fv = zeros(len(W), dtype='d')
        if nonfixed_pairs is None:
            for ref_arc in ref_tree:
                id1, id2, _ = ref_arc
                i = fv_map[id1, id2]
                ref_fv = ref_fv + X[i].toarray()
            for pred_arc in pred_tree:
                id1, id2, _ = pred_arc
                i = fv_map[id1, id2]
                pred_fv = pred_fv + X[i].toarray()
        else:
            for ref_arc in ref_tree:
                id1, id2, _ = ref_arc
                i = fv_map[id1, id2]
                if i in nonfixed_pairs:
                    ref_fv = ref_fv + X[i].toarray()
            for pred_arc in pred_tree:
                id1, id2, _ = pred_arc
                i = fv_map[id1, id2]
                if i in nonfixed_pairs:
                    pred_fv = pred_fv + X[i].toarray()

        # Phi(x,y) - Phi(x,y_hat)
        delta_fv = ref_fv - pred_fv

        # structured loss
        loss_py = float(dot(W, (pred_fv - ref_fv).T))
        # add cost sensitive term
        tloss = self.cost_function(ref_tree, pred_tree, edus)
        # loss_py is not used for the update here, just
        # for the return value to display avg loss
        loss_py += sqrt(tloss)

        # update weights
        if tloss != 0:
            W = W + upd * delta_fv
        self.weights = W
        # print("OUT W:", W)

        # update the average weights
        if self.avg:
            self.avg_weights = self.avg_weights + W

        return loss_py

    def _classify(self, dpack, X, W, nonfixed_pairs=None):
        """ return predicted tree """
        num_items = len(dpack)
        if nonfixed_pairs is None:
            nonfixed_pairs = np.arange(num_items)

        if dpack.graph is None:
            scores = np.zeros(num_items)
            label = np.zeros((num_items, len(dpack.labels)))
            prediction = np.empty(num_items)
        else:
            scores = np.copy(dpack.graph.attach)
            label = np.copy(dpack.graph.label)
            prediction = np.copy(dpack.graph.prediction)

        # compute attachment scores of all EDU pairs
        # TODO should this be self.decision_function?
        # we need to reshape, to lose 2nd dim (shape[1] == 1) of dot product
        scores[nonfixed_pairs] = (X[nonfixed_pairs].dot(W.T)
                                  .reshape(len(nonfixed_pairs)))
        # dummy labelling scores and predictions (for unlabelled parsing)
        unk = dpack.label_number(UNKNOWN)
        # for every pair, set the best label to UNK
        # * score(lbl) = 1.0 if lbl == UNK, 0.0 otherwise
        label[nonfixed_pairs] = 0.0
        label[nonfixed_pairs, unk] = 1.0
        # * predicted label = UNK (will be overwritten by the decoder)
        prediction[nonfixed_pairs] = unk
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

    Parameters
    ----------
    loss : string, optional
        The loss function to be used:
        hinge: PA-I  variant from paper
        squared_hinge: PA-II variant from paper

    cost : string, optional
        The cost function to be used:
        dtree: dependency tree loss
        ctree: constituency tree loss (TODO)
    """

    def __init__(self, decoder,
                 C=1.0,
                 n_iter=5, verbose=0, loss="hinge",
                 cost="dtree",
                 average=False,
                 use_prob=False):
        StructuredPerceptron.__init__(self, decoder,
                                      n_iter=n_iter,
                                      verbose=verbose,
                                      eta0=1.0,
                                      cost=cost,
                                      average=average,
                                      use_prob=use_prob)
        self.C = C
        self.loss = loss

    def update(self, pred_tree, ref_tree, X, fv_map, edus,
               nonfixed_pairs=None):
        r"""PA-II update rule:

        .. math::

            w = w + \tau * (\Phi(x,y)-\Phi(x-\hat{y})) \text{ where}

            \tau = min(C, \frac{loss}{||\Phi(x,y)-\Phi(x-\hat{y})||^2})

            loss = \begin{cases}
                   0             & \text{if } margin \ge 1.0\\
                   1.0 - margin  & \text{otherwise}
                   \end{cases}

            margin =  w \cdot (\Phi(x,y)-\Phi(x-\hat{y}))

        Notes
        -----
        The current implementation corresponds to the prediction-based
        variant of cost sensitive multiclass classification from the
        reference paper.
        We can safely ignore the max-loss variant because it is, if I
        understand correctly, intractable for structured prediction.
        """
        # learning rate, should be defined in fit(), passed to parent fit()
        lr = "pa1" if self.loss == "hinge" else "pa2"
        # end should be in fit()

        W = self.weights
        C = self.C
        # compute Phi(x,y) and Phi(x,y_hat)
        ref_fv = zeros(len(W), dtype='d')
        pred_fv = zeros(len(W), dtype='d')
        if nonfixed_pairs is None:
            for ref_arc in ref_tree:
                id1, id2, _ = ref_arc
                i = fv_map[id1, id2]
                ref_fv = ref_fv + X[i].toarray()
            for pred_arc in pred_tree:
                id1, id2, _ = pred_arc
                i = fv_map[id1, id2]
                pred_fv = pred_fv + X[i].toarray()
        else:
            for ref_arc in ref_tree:
                id1, id2, _ = ref_arc
                i = fv_map[id1, id2]
                if i in nonfixed_pairs:
                    ref_fv = ref_fv + X[i].toarray()
            for pred_arc in pred_tree:
                id1, id2, _ = pred_arc
                i = fv_map[id1, id2]
                if i in nonfixed_pairs:
                    pred_fv = pred_fv + X[i].toarray()

        # Phi(x,y) - Phi(x,y_hat)
        delta_fv = ref_fv - pred_fv
        delta_fv_norm = norm(delta_fv)

        # structured loss
        loss_py = float(dot(W, (pred_fv - ref_fv).T))
        # add cost sensitive term
        tloss = self.cost_function(ref_tree, pred_tree, edus)
        loss_py += sqrt(tloss)

        if lr == "pa1":
            if delta_fv_norm == 0:
                upd = 0
            else:
                upd = delta_fv_norm**2
                upd = min(C, loss_py / upd)
        else:  # "pa2"
            upd = delta_fv_norm**2
            upd = loss_py / (upd + 0.5 / C)

        # update weights
        W = W + upd * delta_fv
        self.weights = W

        # update the average weights
        if self.avg:
            self.avg_weights = self.avg_weights + W

        return loss_py


def _score(w_vect, feat_vect, use_prob=False):
    score = dot(w_vect, feat_vect)
    if use_prob:
        score = expit(score)
    return score
# pylint: enable=no-member
