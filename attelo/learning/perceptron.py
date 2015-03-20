"""
A set of learner variants using a perceptron. The more advanced learners allow
for the possibility of structured prediction.
"""

from __future__ import print_function
import sys
import time
from collections import defaultdict, namedtuple
from numpy.linalg import norm
# pylint: disable=no-name-in-module
# pylint at the time of this writing doesn't deal well with
# packages that dynamically generate methods
# https://bitbucket.org/logilab/pylint/issue/58/false-positive-no-member-on-numpy-imports
from numpy import dot, exp, zeros, sign
from scipy.sparse import csr_matrix  
# pylint: enable-no-name-in-module

from attelo.edu import EDU
from attelo.table import UNLABELLED

# pylint: disable=too-few-public-methods
# pylint: disable=invalid-name
# lots of mathy things here, so names may follow those convetions


"""
TODO:
- add more principled scores to probs conversion (right now, we do just 1-norm
  weight normalization and use logit function)
- add MC perc and PA for relation prediction.
- fold relation prediction into structured learning
"""

# pylint: disable=too-few-public-methods
class PerceptronArgs(namedtuple('PerceptronArgs',
                                ['iterations',
                                 'averaging',
                                 'use_prob',
                                 'aggressiveness'])):
    """
    Parameters for perceptron initialisation

    :param iterations: number of iterations to run
    :type iterations: int > 0

    :param averaging: do averaging on weights
    :type averaging: bool

    :param use_prob: fake a notion of probabilities by using `log` tricks
                     to return scores in [0, 1]
    :type use_prob: bool

    :param aggressiveness: only used for passive-aggressive perceptrons
                           (ignored elsewhere); `inf` gets us a regular
                           perceptron
    :type aggressiveness: float
    """
# pylint: enable=too-few-public-methods


def is_perceptron_model(model):
    """
    If the model in question is somehow based on perceptrons
    """
    return model.name in ["Perceptron",
                          "PassiveAggressive",
                          "StructuredPerceptron",
                          "StructuredPassiveAggressive"]

class Perceptron(object):
    """ Vanilla binary perceptron learner """
    def __init__(self, pconfig):
        self.name = "Perceptron"
        self.nber_it = pconfig.iterations
        self.avg = pconfig.averaging
        self.use_prob = pconfig.use_prob
        self.weights = None
        self.avg_weights = None
        return
    
    def fit(self, X, Y): # X contains all EDU pairs for corpus
        """ learn perceptron weights """
        self.init_model( X ) 
        self.learn( X, Y ) 
        return self

    def predict(self, X): 
        W = self.avg_weights if self.avg else self.weights
        return sign( X.dot(W.T) )

    def decision_function(self, X):
        W = self.avg_weights if self.avg else self.weights
        return X.dot(W.T)
    
    def init_model(self, X):
        dim = X.shape[1]
        print("FEAT. SPACE SIZE:",dim)
        self.weights = zeros(dim, 'd')
        self.avg_weights = zeros(dim, 'd')
        return

    def learn(self, X, Y):
        start_time = time.time()
        print("-"*100, file=sys.stderr)
        print("Training...", file=sys.stderr)
        nber_it = self.nber_it
        for n in xrange(nber_it):
            print("it. %3s \t" % n, file=sys.stderr)
            loss = 0.0
            t0 = time.time()
            inst_ct = 0
            for i in xrange(X.shape[0]):
                X_i = X[i]
                Y_i = Y[i]
                inst_ct += 1
                sys.stderr.write("%s" %"\b"*len(str(inst_ct))+str(inst_ct))
                Y_hat, score = self._classify(X_i, self.weights)
                loss += self.update(Y_hat, Y_i, X_i, score)
            if inst_ct > 0:
                loss = loss / float(inst_ct)
            t1 = time.time()
            print("\tavg loss = %-7s" % round(loss, 6), file=sys.stderr)
            print("\ttime = %-4s" % round(t1-t0, 3), file=sys.stderr)
        elapsed_time = t1-start_time
        print("done in %s sec." % round(elapsed_time, 3), file=sys.stderr)
        return


    def update(self, Y_j_hat, Y_j, X_j, score, rate=1.0):
        """ simple perceptron update rule"""
        X_j = X_j.toarray()
        error = (Y_j_hat != Y_j)
        W = self.weights
        if error:
            W = W + rate * Y_j * X_j
            self.weights = W
        if self.avg:
            self.avg_weights = self.avg_weights +  W
        return int(error)


    def _classify(self, X, W):
        """ classify feature vector X using weight vector w into
        {-1,+1}"""
        score = float( X.dot(W.T) )
        return sign(score), score







class PassiveAggressive(Perceptron):
    """
    Passive-Aggressive classifier in primal form. PA has a
    margin-based update rule: each update yields at least a margin
    of one (see defails below). Specifically, we implement PA-II
    rule for the binary setting (see Crammer et. al 2006). Default
    C=inf parameter makes it equivalent to simple PA.
    """

    def __init__(self, pconfig):
        Perceptron.__init__(self, pconfig)
        self.name = "PassiveAggressive"
        self.aggressiveness = pconfig.aggressiveness
        return


    def update(self, Y_j_hat, Y_j, X_j, score):
        r"""PA-II update rule

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
        C = self.aggressiveness
        margin = Y_j * score
        loss = 0.0
        if margin < 1.0:
            loss = 1.0-margin
        norme = norm(X_j)
        if norme != 0:
            tau = loss / float(norme**2)
        tau = min(C, tau)
        W = W + tau * Y_j * X_j
        self.weights = W
        if self.avg:
            self.avg_weights = self.avg_weights +  W
        return loss




class StructuredPerceptron(Perceptron):
    """ Perceptron classifier (in primal form) for structured
    problems."""


    def __init__(self, decoder, pconfig):
        Perceptron.__init__(self, pconfig)
        self.name = "StructuredPerceptron"
        self.decoder = decoder
        return

    def init_model(self, dim):
        print("FEAT. SPACE SIZE:",dim)
        self.weights = zeros(dim, 'd')
        self.avg_weights = zeros(dim, 'd')
        return

    def fit_structured(self, datapacks, _targets): # datapacks is an datapack iterable
        """ learn struct. perceptron weights """        
        self.init_model( datapacks[0].data.shape[1] )
        self.learn( datapacks ) 
        return self

    def learn(self, datapacks):
        start_time = time.time()
        print("-"*100, file=sys.stderr)
        print("Training struct. perc...", file=sys.stderr)
        for n in range(self.nber_it):
            print("it. %3s \t" % n, file=sys.stderr)
            loss = 0.0
            t0 = time.time()
            inst_ct = 0
            for dpack in datapacks:
                inst_ct += 1
                sys.stderr.write("%s" %"\b"*len(str(inst_ct))+str(inst_ct))
                # extract data and target
                X = dpack.data # each row is EDU pair
                Y = dpack.target # each row is {-1,+1}
                # construct ref graph and mapping {edu_pair => index in X}
                edu_pairs = dpack.pairings
                ref_tree = []
                fv_index_map = {}
                for i,(id1,id2) in enumerate(edu_pairs):
                    fv_index_map[id1,id2] = X[i]
                    if Y[i] == 1:
                        ref_tree.append( (id1, id2, UNLABELLED) )
                # predict tree based on current weight vector
                pred_tree = self._classify(X, edu_pairs, self.weights)
                # print doc_id,  predicted_graph
                loss += self.update(pred_tree, ref_tree, fv_index_map)
            # print(inst_ct,, file=sys.stderr)
            avg_loss = loss / float(inst_ct)
            t1 = time.time()
            print("\tavg loss = %-7s" % round(avg_loss, 6), file=sys.stderr)
            print("\ttime = %-4s" % round(t1-t0, 3), file=sys.stderr)
        elapsed_time = t1-start_time
        print("done in %s sec." % round(elapsed_time, 3), file=sys.stderr)
        return

    def update(self, pred_tree, ref_tree, fv_map, rate=1.0):
        # rt = [(t[0].span(),t[1].span()) for t in ref_tree]
        # pt = [(t[0].span(),t[1].span()) for t in pred_tree]
        # print("REF TREE:", rt)
        # print("PRED TREE:", pt)
        # print("INTER:", set(pt) & set(rt))
        W = self.weights
        # print("IN W:", W)
        # error = not( set(pred_tree) == set(ref_tree) )
        loss = tree_loss( ref_tree, pred_tree )
        if loss != 0:
            ref_fv = zeros(len(W), 'd')
            pred_fv = zeros(len(W), 'd')
            for ref_arc in ref_tree:
                id1, id2, _ = ref_arc
                ref_fv = ref_fv + fv_map[id1, id2].toarray()
            for pred_arc in pred_tree:
                id1, id2, _ = pred_arc
                pred_fv = pred_fv + fv_map[id1, id2].toarray()
            W = W + rate * (ref_fv - pred_fv)
        # print("OUT W:", W)
        self.weights = W
        if self.avg:
            self.avg_weights = self.avg_weights + W
        return loss


    def _classify(self, X, edu_pairs, W):
        """ return predicted tree """
        decoder = self.decoder
        scores = X.dot(W.T)
        scored_tuples = []
        for i,(id1, id2) in enumerate(edu_pairs):
            scored_tuples.append((EDU(id1, 0, 0, None, None, None), # hacky 
                                  EDU(id2, 0, 0, None, None, None),
                                  scores[i],
                                  UNLABELLED))
        # print "SCORES:", scores
        pred_tree = decoder.decode(scored_tuples)[0]
        return pred_tree




class StructuredPassiveAggressive(StructuredPerceptron):
    """Structured PA-II classifier (in primal form) for structured
    problems."""


    def __init__(self, decoder, pconfig):
        StructuredPerceptron.__init__(self, decoder, pconfig)
        self.name = "StructuredPassiveAggressive"
        self.aggressiveness = pconfig.aggressiveness
        return


    def update(self, pred_tree, ref_tree, fv_map):
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
        C = self.aggressiveness
        # compute Phi(x,y) and Phi(x,y^)
        ref_fv = zeros(len(W), 'd')
        pred_fv = zeros(len(W), 'd')
        for ref_arc in ref_tree:
            id1, id2, _ = ref_arc
            ref_fv = ref_fv + fv_map[id1, id2].toarray()
        for pred_arc in pred_tree:
            id1, id2, _ = pred_arc
            pred_fv = pred_fv + fv_map[id1, id2].toarray()
        # find tau
        delta_fv = ref_fv-pred_fv
        margin = float(dot(W, delta_fv.T))
        loss = 0.0
        tau = 0.0
        if margin < 1.0:
            loss = 1.0-margin
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



def tree_loss(ref_tree, pred_tree):
    return 1.0 - (len(set(pred_tree) & set(ref_tree))/ float(len(ref_tree)))


def _score(w_vect, feat_vect, use_prob=False):
    score = dot(w_vect, feat_vect)
    if use_prob:
        score = logit(score)
    return score


def logit(score):
    """ return score in [0,1], i.e., fake probability"""
    return 1.0/(1.0+exp(-score))
# pylint: enable=no-member
