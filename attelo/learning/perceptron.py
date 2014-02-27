import sys
import time
from collections import defaultdict, namedtuple
from numpy import *

from attelo.edu import EDU, mk_edu_pairs

"""
TODO:
- use_prob in constructor of StructurePerc
- separate perc code from orange interface
- handle with redundant features
- feature counts and cutoff low freqs
- allow decoder parameter for struct perc
- aggressive updates
- feature combinations or kernels
- integrate relation prediction.
"""

PerceptronArgs = namedtuple('PerceptronArgs', 'iterations averaging use_prob')

"""
REMINDER:

functions to expose:
- __call__( orange_data )
- get_probs( doc_instances )

"""


class OrangeInterface( object ):

    def __init__(self, data, meta_features):
        self.__data = data
        self.__meta_features = meta_features
        self.set_feature_map()
        return

    def set_feature_map( self ): # FIXME: remove redundant features: True/False
        """ binarizing all features and construct feature-to-integer map """
        domain = self.__domain = self.__data.domain
        fmap = {} 
        pos = 0
        print >> sys.stderr, "# of orange features", len(domain.features)
        for feat in domain.features:
            if str(feat.var_type) == "Continuous":
                fmap[feat.name] = pos
                pos += 1
            elif str(feat.var_type) == "Discrete":
                for val in feat.values:
                    fmap[feat.name,val] = pos
                    pos += 1
            else:
                raise TypeError("Unsupported orange feature type: %s" %feat.var_type)
        print >> sys.stderr, "# of binarized features", len(fmap)
        self.__feature_map = fmap
        return


    def get_feature_map( self ):
        return self.__feature_map


    def get_edu_pair(self, orange_inst):
        return mk_edu_pairs(self.__meta_features, self.__domain)(orange_inst)
    

    def instance_convertor( self, orange_inst ):
        """ convert orange instance into feature vector """
        fmap = self.__feature_map
        fv = zeros( len(fmap) )
        classe = None
        edu_pair = self.get_edu_pair( orange_inst )            
        for av in orange_inst:
            att_name = av.variable.name
            att_type = str(av.var_type)
            att_val = av.value
            # get class label (do not use it as feature :-))
            if att_name == self.__meta_features.label: 
                if av.value == "True":
                    classe = 1
                elif av.value == "False":
                    classe = -1
                else:
                    raise NotImplementedError("Only binary classes for now!")
            else:
                # build feature vector by looking up the feature map
                if att_type == "Continuous":
                    fv[fmap[att_name]] = att_val
                elif att_type == "Discrete":
                    try:
                        fv[fmap[att_name,att_val]] = 1.0
                    except KeyError:
                        pass
                        # print >> sys.stderr, "Unseen feature:", (att_name,att_val) 
                else:
                    raise TypeError("Unknown feature type/value: '%s'/'%s'" %(att_type,av.value))
        assert classe in [-1,1], "label (%s) not in {-1,1}" %classe
        return edu_pair, classe, fv


    def train_instance_generator( self ):
        return self.instance_generator( self.__data )
    
    
    def instance_generator( self, data ):
        for instance in data:
            yield self.instance_convertor( instance )

            


class Perceptron( object ):
    """ Vanilla binary perceptron learner """
    def __init__(self, meta_features, nber_it=1, avg=False):
        self.name = "Perceptron"
        self.__meta_features = meta_features
        self.__nber_it = nber_it
        self.__avg = avg
        self.__weights = None
        self.__avg_weights = None
        self.__orange_interface = None
        return
    
    
    def __call__(self, orange_train_data):
        """ learn perceptron weights """
        interface = OrangeInterface( orange_train_data, self.__meta_features )
        train_instances = interface.train_instance_generator()
        self.__init_model( interface.get_feature_map() )
        # self.__learn( train_instances ) 
        self.__orange_interface = interface
        return self


    def get_probs( self, doc_orange_instances ):
        """ return scores obtained for instances with learned weights """
        interface = self.__orange_interface
        doc_instances = interface.instance_generator( doc_orange_instances )
        return self.__get_scores( doc_instances, use_prob=True )


    def __init_model( self, feature_map ):
        dim = len( feature_map )
        self.__weights = zeros( dim, 'd' )
        self.__avg_weights = zeros( dim, 'd' )
        return


    def __learn( self, instances ):
        start_time = time.time()
        print >> sys.stderr, "-"*100
        print >> sys.stderr, "Training..."
        nber_it = self.__nber_it
        for n in range( nber_it ):
            print >> sys.stderr, "it. %3s \t" %n, 
            loss = 0.0
            t0 = time.time()
            inst_ct = 0
            for _, ref_cl, fv in instances:
                inst_ct += 1
                sys.stderr.write("%s" %"\b"*len(str(inst_ct))+str(inst_ct))
                pred_cl, _ = self.__classify( fv, self.__weights )
                loss += self.__update( pred_cl, ref_cl, fv )
            avg_loss = loss / float(inst_ct)
            t1 = time.time()
            print >> sys.stderr, "\tavg loss = %-7s" %round(avg_loss,6),
            print >> sys.stderr, "\ttime = %-4s" %round(t1-t0,3)
        elapsed_time = t1-start_time
        print >> sys.stderr, "done in %s sec." %(round(elapsed_time,3))
        return

    def __update( self, pred, ref, fv, rate=1.0 ): 
        """ simple perceptron update rule"""
        error = (pred != ref)
        w = self.__weights
        if error:
            w = w + rate * ref * fv
            self.__weights = w
        if self.__avg:
            self.__avg_weights += w
        return int(error)


    def __classify( self, fv, w ):
        """ classify feature vector fv using weight vector w into
        {-1,+1}"""
        score = dot( w, fv )
        label = 1 if score >= 0 else -1
        return label, score


    def __get_scores( self, doc_instances, use_prob=False ):
        scores = []
        w = self.__avg_weights if self.__avg else self.__weights
        for edu_pair, _, fv in doc_instances:
            _, score = self.__classify( fv, w )
            # print "\t", edu1, edu2, pred_cl, score
            if use_prob:
                # logit
                score = 1.0/(1.0+exp(-score)) 
            scores.append( (edu_pair[0], edu_pair[1], score, "unlabelled") )
        return scores








class StructuredPerceptron( Perceptron ):
    """ Perceptron classifier (in primal form) for structured
    problems.""" 


    def __init__( self, features, decoder, nber_it=1, avg=False ):
        Perceptron.__init__(self, features, nber_it=nber_it, avg=avg)
        self.name = "StructuredPerceptron"
        self._decoder = decoder 
        self._use_prob = use_prob
        return
    

    def __call__(self, train_orange_data_table):
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
            doc_name = edu_pair_inst[self.features.grouping].value
            edu1, edu2, cl, fv = self.unpack_orange( edu_pair_inst, fv_fct )
            doc2fvs[doc_name][edu1.id,edu2.id] = fv
            if cl == 1:
                doc2ref_graph[doc_name].append( (edu1.id, edu2.id, "unlabelled") )
        print >> sys.stderr, "done."
        # learn from feature vectors
        print >> sys.stderr, "Learning..."
        self.learn( doc2fvs, doc2ref_graph, use_prob=self._use_prob )
        print >> sys.stderr, "done."
        return self 


    def learn( self, doc2fvs, doc2ref_graph ):
        """ update model paramater vector in a round-like fashion
        based on comparison between the outcome predicted by current
        parameter vector and true outcome"""
        start_time = time.time()
        print >> sys.stderr, "-"*100
        print >> sys.stderr, "Training struct. perc..." 
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
                                                 use_prob=self._use_prob) 
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


    def classify( self, fvs, weights ):
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
        pred_graph = decoder( scores, use_prob=self._use_prob )
        return pred_graph


    # def predict( self, doc_instances ):
    #     fv_fct = self.feature_vector
    #     fvs = {}
    #     for one in doc_instances:
    #         edu1, edu2, _, fv = self.unpack_orange( one, fv_fct )
    #         fvs[edu1.id,edu2.id] = fv
    #     w = self._avg_weights if self._avg else self._weights
    #     return self.classify( fvs, w )
    

    def get_scores(self, doc_instances): # get local scores
        fv_fct = self.feature_vector
        w = self._avg_weights if self._avg else self._weights
        scores = []
        for one in doc_instances:
            edu1, edu2, _, fv = self.unpack_orange( one, fv_fct )
            score = dot( w, fv )
            if self._use_prob:
                # logit
                score = 1.0/(1.0+exp(-score))
            scores.append( (edu1, edu2, score, "unlabelled" ) )
        return scores



  



    




