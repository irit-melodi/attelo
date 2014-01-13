"""
Experiment results
"""

import os
import sys
import cPickle

try:
    STATS=True
    from scipy.stats import wilcoxon, ttest_rel, mannwhitneyu, sem, bayes_mvs
except:
    print >> sys.stderr, "no module scipy.stats, cannot test stat. significance of results"
    STATS=False

def f1_score(p_r):
    (prec,recall) = p_r
    return 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0

class Report:
    """class holding experiments results, for nice output (not implemented)

    todo: ? correction built-in the saved scores or saved ? 
    """
    def __init__(self, evals, params = None, correction = 1.0):
        correct, total_pred, total_ref = map(lambda x:sum(x), zip(*evals))
        all_correct, all_pred, all_total = zip(*evals)
        self._measures = {}
        self._measures["prec"] = zip(all_correct,all_pred)
        self._measures["recall"] = zip(all_correct,all_total)
        # behold the power of functional programming !
        prec_all = map(lambda (x,y): x/float(y) if y> 0 else 0,self._measures["prec"])
        rec_all = map(lambda (x,y): x/float(y) if y> 0 else 0,self._measures["recall"])
        self._measures["F1"] = map(f1_score,zip(prec_all,rec_all))

        prec = float(correct) / total_pred if total_pred > 0 else 0
        recall = float(correct) / total_ref if total_ref > 0 else 0
        self.f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0
        self.prec = prec
        self.recall = recall
        self.correct = correct
        self.total_pred = total_pred
        self.total_ref = total_ref
        self.correction = correction
        self.params = params
        if self.correction < 1.0:
            corr_r = self.recall * self.correction
            self.f1_corr = (2 * prec * corr_r / (prec + corr_r)) if (prec + corr_r) > 0 else 0
            self.corr_r = corr_r
        else:
            self.f1_corr = None
            self.corr_r = None

    def table(self, mode = "latex"):
        # plain, latex
        pass

    def save(self,name):
        dname = os.path.dirname(name)
        if not os.path.exists(dname):
            os.makedirs(dname)
        reportfile=open(name,"w")
        cPickle.dump(self,reportfile)
        reportfile.close()


    def plot(self, mode = "roc"):
        # plot curve:  learning curve, nbest solutions, what else ?
        pass

    def summary(self):
        output = ["{relations} {context} ({relnb}) : \t {decoders}+{learners}, h={heuristics}, unlabelled={unlabelled},post={post_label},rfc={rfc}".format(**self.params.__dict__)]
        prec, recall, f1 = self.prec, self.recall, self.f1
        try:
            m,(a,b) = self.confidence_interval()
        except:
            print >> sys.stderr, "warning: not able to compute confidence interval"
            m, a = (0,0)
        output.append( "\t Prec=%1.3f, Recall=%1.3f, F1=%1.3f +/- %1.3f (%1.3f +- %1.3f)" % (prec, recall, f1,self.standard_error(),m,(m-a)))
        if self.f1_corr:
            output.append( "\t with recall correction estimate, R=%1.3f, F1=%1.3f" % (self.corr_r, self.f1_corr))
        return " ".join(output)

    def standard_error(self,measure="F1"):
        """standard error (of the mean) on measures by text (TODO: folds should be saved too ?)
        """
        return sem(self._measures[measure])

    def confidence_interval(self,measure="F1",alpha=0.95):
        # will return mean, ci 
        return bayes_mvs(self._measures[measure],alpha)[0]


    # inspired by Apetite Evaluation class
    def significance(self,other,measure="F1",test="wilcoxon"):
        """computes stats significance of difference between two sets of scores
        test can be paired wilcoxon, mannwhitney for indep samples, or paired ttest. 
        """
        d1 = self._measures[measure]
        d2 = other._measures[measure]
        if type(d1[0])==type(1.0) or type(d1[0])==type(1):
            pass
        else:
            d1 = [x for (x,y) in d1]
            d2 = [x for (x,y) in d2]
          
        differences = [(x,y) for (x,y) in zip(d1,d2) if x!=y] 
        #print >> sys.stderr, differences
        #print >> sys.stderr, d2
        #print >> sys.stderr, [x for (i,x) in enumerate(d1) if x!=d2[i]]
        assert len(d1)==len(d2)           
        
        results = {}
            
        if test=="wilcoxon" or test == "all":
            tscore,p = wilcoxon(d1,d2)
            results["wilcoxon"] = p
        if test == "ttest" or  test == "all":
            tscore,p = ttest_rel(d1,d2)
            results["paired ttest"] = p
        if test =="mannwhitney" or  test == "all":
            tscore,p = mannwhitneyu(d1,d2)
            results["mannwhitney"] = p 
        return results
