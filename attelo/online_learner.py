import sys
import time
from collections import defaultdict
from numpy import *
from edu import EDU

# meta-features used for nfold, indexing, etc
# default is for annodis coling experiments. 
# best to pass on as argument to constructor for learners
def_config = {
"FirstNode" : "SOURCE",
"SecondNode" : "TARGET",
"TargetSpanStart" : "TargetSpanStart",
"TargetSpanEnd" : "TargetSpanEnd",
"SourceSpanStart" : "SourceSpanStart",
"SourceSpanEnd" : "SourceSpanEnd",
"FILE" : "FILE"
}


"""
TODO:
- allow decoder parameter: done.
- aggressive updates
- feature combinations or kernels
- integrate relation prediction.
"""


class Perceptron( object ):

    def __init__( self, domain=None, nber_it=1, avg=False, cfg = def_config ):
        self.name = "Perceptron"
        self._weights = None
        self._avg_weights = None
        self._domain = domain
        if domain:
            self._set_domain( domain )
        self._avg = avg
        self._nber_it = nber_it
        self._cfg = cfg
        return

    def set_domain(self, domain):
        self._domain = domain
        self._weights = zeros( len(domain.attributes), 'd' )
        self._avg_weights = zeros( len(domain.attributes), 'd' )
        return

    def get_cfg(self,option):
        return self._cfg.get(option)

    def get_current_weights(self):
        """ return current weights """
        return self._get_weights(avg=False)

    def get_avg_weights(self):
        """ return averaged weights """
        return self._get_weights(avg=True)

    def _get_weights(self, avg=False):
        w = self._get_weights(avg=avg)
        return w

    def _get_weights(self, avg=False):
        """ return current weights or averaged weights """
        weights = self._weights
        if avg == True:
            weights = self._avg_weights
        return weights


    def edu_pair_from_orange( self, orange_instance):
        domain = self._domain
        arg1 = domain.index(self.get_cfg("FirstNode"))
        arg2 = domain.index(self.get_cfg("SecondNode"))
        TargetSpanStartIndex = domain.index(self.get_cfg("TargetSpanStart"))
        TargetSpanEndIndex = domain.index(self.get_cfg("TargetSpanEnd"))
        SourceSpanStartIndex = domain.index(self.get_cfg("SourceSpanStart"))
        SourceSpanEndIndex = domain.index(self.get_cfg("SourceSpanEnd"))
        FILEIndex = domain.index(self.get_cfg("FILE"))
        edu1 = EDU(orange_instance[arg1].value, orange_instance[SourceSpanStartIndex].value,\
                   orange_instance[SourceSpanEndIndex].value, orange_instance[FILEIndex].value)
        edu2 = EDU(orange_instance[arg2].value, orange_instance[TargetSpanStartIndex].value, \
                   orange_instance[TargetSpanEndIndex].value, orange_instance[FILEIndex].value)
        return edu1, edu2


    def fv_from_orange( self, orange_instance, fv_fct ):
        attr_val_list = []
        classe = None
        for av in orange_instance:
            att_name = av.variable.name
            att_type = str(av.var_type)
            if att_name == "CLASS": # do not use CLASS as feature :-)
                classe = 1 if str(av.value) == "True" else -1
            else:
                if att_type == "Continuous":
                    att_val = av.value
                elif av.value in ["True", "true"]: # FIXME: super ad hoc
                    att_val = 1.0
                elif av.value in ["False", "false"]:
                    att_val = 0.0
                else:
                    raise TypeError("Unknown feature type/value: '%s'/'%s'" %(att_type,av.value))
                attr_val_list.append( (att_name,att_val) )
        assert classe in [-1,1]
        return fv_fct( attr_val_list ), classe


    def unpack_orange( self, orange_instance, fv_fct ):
        edu1, edu2 = self.edu_pair_from_orange( orange_instance )
        fv, cl = self.fv_from_orange( orange_instance, fv_fct ) 
        return edu1, edu2, cl, fv


    def feature_vector(self, attr_val_list):
        """ build feature vector from attribute-value list by looking
        up domain index"""
        domain = self._domain
        fv = zeros( len(domain.attributes), 'd' )
        for feat,val in attr_val_list: 
            pos = domain.index( feat )
            try:
                fv[pos] = float(val)
            except ValueError:
                sys.exit("FeatureVector Error: non float (%s) value found!" %type(val))
        return fv


    def __call__(self, train_orange_data_table):
        # set domain for feature vector creation
        domain = train_orange_data_table.domain
        self.set_domain( domain )
        # learn from instances
        self.learn( train_orange_data_table ) 
        return self 

    
    def learn( self, instances ):
        """ update model paramater vector in a round-like fashion
        based on comparison between the outcome predicted by current
        parameter vector and true outcome"""
        start_time = time.time()
        print >> sys.stderr, "-"*100
        print >> sys.stderr, "Training..."
        nber_it = self._nber_it
        fv_fct = self.feature_vector
        # caching for multiple iterations
        edu_pair2fv = {}
        for n in range( nber_it ):
            print >> sys.stderr, "it. %3s \t" %n, 
            loss = 0.0
            t0 = time.time()
            inst_ct = 0
            for one in instances:
                inst_ct += 1
                sys.stderr.write("%s" %"\b"*len(str(inst_ct))+str(inst_ct))
                # edu1, edu2, ref_cl, fv = self.unpack_orange( one, fv_fct )
                if n == 0:
                    edu1, edu2, ref_cl, fv = self.unpack_orange( one, fv_fct )
                    # caching for next iterations
                    edu_pair2fv[edu1.id,edu2.id] = (fv, ref_cl)
                else:
                    edu1, edu2 = self.edu_pair_from_orange( one )
                    fv, ref_cl = edu_pair2fv[edu1.id,edu2.id]
                pred_cl, _ = self.classify( fv, self._weights )
                # print "\t", edu1, edu2, pred_cl == ref_cl
                loss += self.update( pred_cl, ref_cl, fv )
            # print >> sys.stderr, inst_ct,
            avg_loss = loss / float(inst_ct)
            t1 = time.time()
            print >> sys.stderr, "\tavg loss = %-7s" %round(avg_loss,6),
            print >> sys.stderr, "\ttime = %-4s" %round(t1-t0,3)
        elapsed_time = t1-start_time
        print >> sys.stderr, "done in %s sec." %(round(elapsed_time,3))
        return

        
    def update( self, pred, ref, fv, rate=1.0 ): 
        """ simple perceptron update rule"""
        error = (pred != ref)
        w = self._weights
        if error:
            w = w + rate * ref * fv
            self._weights = w
        if self._avg:
            self._avg_weights += w
        return int(error)


    def classify( self, fv, w ):
        """ classify feature vector fv using weight vector w into
        {-1,+1}"""
        # print instance
        score = dot( w, fv )
        label = 1 if score >= 0 else -1
        return label, score


    def get_scores( self, doc_instances, use_prob=False ):
        fv_fct = self.feature_vector
        scores = []
        w = self._avg_weights if self._avg else self._weights
        for one in doc_instances:
            edu1, edu2, _ref_cl, fv = self.unpack_orange( one, fv_fct ) 
            _, score = self.classify( fv, w )
            # print "\t", edu1, edu2, pred_cl, score
            if use_prob:
                # logit
                score = 1.0/(1.0+exp(-score)) 
            scores.append( (edu1, edu2, score, "unlabelled") )
        return scores


    def get_probs( self, doc_instances ):
        return self.get_scores( doc_instances, use_prob=True )


    # def predict( self, doc_instances ): # FIXME: not needed (use local decoder instead!)
    #     """ local decoding """
    #     fv_fct = self.feature_vector
    #     pred_graph = []
    #     w = self._avg_weights if self._avg else self._weights
    #     for one in doc_instances:
    #         edu1, edu2, _ref_cl, fv = self.unpack_orange( one, fv_fct ) 
    #         pred_cl, _score = self.classify( fv, w )
    #         # print "\t", edu1, edu2, pred_cl, _ref_cl, _score
    #         if pred_cl == 1:
    #             pred_graph.append( (edu1.id, edu2.id, "unlabelled" ) )
    #     return pred_graph       








class StructuredPerceptron( Perceptron ):
    """ Perceptron classifier (in primal form) for structured
    problems.""" 

    def __init__( self, decoder, domain=None, nber_it=1, avg=False, cfg = def_config ):
        Perceptron.__init__(self, domain=domain, nber_it=nber_it, avg=avg, cfg = cfg)
        self.name = "StructuredPerceptron"
        self._decoder = decoder 
        return
    

    def __call__(self, train_orange_data_table, use_prob=False):
        print >> sys.stderr, "Instance conversion...",
        # set domain for feature vector creation
        domain = train_orange_data_table.domain
        self.set_domain( domain )
        # group instances by documents and build document reference
        # graph
        fv_fct = self.feature_vector
        doc2fvs = defaultdict(dict)
        doc2ref_graph = defaultdict(list)
        # edu_id2edu = {}
        for edu_pair_inst in train_orange_data_table:
            doc_name = edu_pair_inst[self.get_cfg("FILE")].value
            edu1, edu2, cl, fv = self.unpack_orange( edu_pair_inst, fv_fct )
            doc2fvs[doc_name][edu1.id,edu2.id] = fv
            if cl == 1:
                doc2ref_graph[doc_name].append( (edu1.id, edu2.id, "unlabelled") )
        print >> sys.stderr, "done."
        # learn from feature vectors
        print >> sys.stderr, "Learning..."
        self.learn( doc2fvs, doc2ref_graph, use_prob=use_prob )
        print >> sys.stderr, "done."
        return self 


    def learn( self, doc2fvs, doc2ref_graph, use_prob=False ):
        """ update model paramater vector in a round-like fashion
        based on comparison between the outcome predicted by current
        parameter vector and true outcome"""
        start_time = time.time()
        print >> sys.stderr, "-"*100
        print >> sys.stderr, "Training..." 
        for n in range( self._nber_it ):
            print >> sys.stderr, "it. %3s \t" %n, 
            loss = 0.0
            t0 = time.time()
            inst_ct = 0
            for doc_id, fvs in doc2fvs.items():
                inst_ct += 1
                sys.stderr.write("%s" %"\b"*len(str(inst_ct))+str(inst_ct))
                # make prediction based on current weight vector
                predicted_graph = self.classify( fvs,
                                                 self._weights, # use current weight vector
                                                 use_prob=use_prob) 
                # print doc_id,  predicted_graph 
                loss += self.update( predicted_graph,
                                     doc2ref_graph[doc_id],
                                     fvs)
            # print >> sys.stderr, inst_ct, 
            avg_loss = loss / float(inst_ct)
            t1 = time.time()
            print >> sys.stderr, "\tavg loss = %-7s" %round(avg_loss,6),
            print >> sys.stderr, "\ttime = %-4s" %round(t1-t0,3)
        elapsed_time = t1-start_time
        print >> sys.stderr, "done in %s sec." %(round(elapsed_time,3))
        return

        
    # def update( self, pred_graph, ref_graph, fvs, rate=1.0 ): 
    #     # print "REF GRAPH:", sorted(ref_graph)
    #     # print "PRED GRAPH:", sorted(pred_graph)
    #     w = self._weights        
    #     # print "W in:", w
    #     fn_ct = 0
    #     for arc in ref_graph:
    #         edu1_id, edu2_id, _ = arc
    #         if arc not in pred_graph:
    #             fn_ct += 1
    #             w = w + rate * fvs[edu1_id, edu2_id]
    #     fp_ct = 0
    #     for arc in pred_graph:
    #         edu1_id, edu2_id, _ = arc
    #         if arc not in ref_graph:
    #             fp_ct += 1
    #             w = w - rate * fvs[edu1_id, edu2_id]
    #     error = fn_ct + fp_ct
    #     if self._avg:
    #         self._avg_weights += w
    #     # print "W out:", w
    #     self._weights = w
    #     return int(error)


    def update( self, pred_graph, ref_graph, fvs, rate=1.0 ): 
        # print "REF GRAPH:", sorted(ref_graph)
        # print "PRED GRAPH:", sorted(pred_graph)
        w = self._weights
        # print "W in:", w
        error = (set(pred_graph) != set(ref_graph))
        if error:
            domain = self._domain
            ref_global_fv = zeros( len(domain.attributes), 'd' )
            pred_global_fv = zeros( len(domain.attributes), 'd' )
            for arc in ref_graph:
                edu1_id, edu2_id, _ = arc
                ref_global_fv = ref_global_fv + fvs[edu1_id, edu2_id]
            for arc in pred_graph:
                edu1_id, edu2_id, _ = arc
                pred_global_fv = pred_global_fv + fvs[edu1_id, edu2_id]
            w = ( ref_global_fv - pred_global_fv )
        if self._avg:
            self._avg_weights += w
        # print "W out:", w
        self._weights = w
        return int(error)


    def classify( self, fvs, weights, use_prob=False ):
        """ return predicted graph """
        decoder = self._decoder
        scores = []
        for (edu1_id, edu2_id), fv in fvs.items():
            score = dot( weights, fv )
            # print "\t", edu1_id, edu2_id, score # , fv
            if use_prob:
                # logit
                score = 1.0/(1.0+exp(-score))
            scores.append( ( EDU(edu1_id, 0, 0, None), # hacky
                             EDU(edu2_id, 0, 0, None),
                             score,
                             "unlabelled" ) )
        # print "SCORES:", scores
        pred_graph = decoder( scores, use_prob=use_prob )
        return pred_graph


    # def predict( self, doc_instances ):
    #     fv_fct = self.feature_vector
    #     fvs = {}
    #     for one in doc_instances:
    #         edu1, edu2, _, fv = self.unpack_orange( one, fv_fct )
    #         fvs[edu1.id,edu2.id] = fv
    #     w = self._avg_weights if self._avg else self._weights
    #     return self.classify( fvs, w )
    

    def get_scores( self, doc_instances, use_prob=False ): # get local scores
        fv_fct = self.feature_vector
        w = self._avg_weights if self._avg else self._weights
        scores = []
        for one in doc_instances:
            edu1, edu2, _, fv = self.unpack_orange( one, fv_fct )
            score = dot( w, fv )
            if use_prob:
                # logit
                score = 1.0/(1.0+exp(-score))
            scores.append( (edu1, edu2, score, "unlabelled" ) )
        return scores



  



    




