#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TODO:
    - refactor to be a FeatureMap API.
    then script should accept commands

    - refactor feature extraction to be built on pairs of edus, with API from annodisReader

read feature map file, and add some feature
    x- lexical neighbours provided with sqlite db
    - deal with complex segments too
    x- anaphoric features

must be provided with original input file for document in xml and feature file
and lexical similarity database

eg

python add_features.py WK377-Omerta -v voisins.db -t -a

and must be present:
WK377-Omerta.xml
WK377-Omerta.features
WK377-Omerta.txt.prep.xml

ajouts:
cd ../../../data/attachment/dev/
for f in *.features; do python ../../../code/src/attachement/add_feature.py `basename $f .features` -b -c -t -a -v /home/phil/Devel/Voiladis/data/voisins.db ; done;

fusion:
cd ../../../data/attachment/dev/
python ../../../code/src/attachement/add_feature.py -m features -f full -o all.csv

selection of features (-l file keep only feature listed in file):
for f in *.features; do python ../../../code/src/attachement/add_feature.py `basename $f .features` -l xxx -o ../dev-select/$f ; done;

selection of instances: eg. keep only valid attachment, between EDUs
for f in *.features; do python ../../../code/src/attachement/add_feature.py `basename $f .features` -i "D#N_EDU=True,D#C_EDU=True,c#CLASS=True" -o ../select/$f ; done;

All Options:
  -h, --help            show this help message and exit
  -t, --timex           add timex (def: False)
  -c, --verb-class      add event verb class (def: False)
  -a, --anaphor         add anaphor (def: False)
  -v VOISINS, --voisins=VOISINS
                        lex neighbors DB
  -w, --weird           if input is in weird format for class attribute
  -d, --distance        discretize distance
  -f FORMAT, --format=FORMAT
                        output format (sparse of full); sanity outputs the
                        reference, just for verification
  -o OUTPUT, --output=OUTPUT
                        output destination (default: inplace modification)
  -m MERGE, --merge=MERGE
                        merge all features files with given suffix into one
                        big output;                            do not try to
                        add features at the same time (undefined results will
                        bite you in the )!
  -b, --binarise        binarise discrete non boolean features
  -l FILTER, --filter=FILTER
                        use given file to filter out features
  -k, --kill            kill selected features instead of keeping them (def:
                        false)
  -i INST_FILTER, --inst-filter=INST_FILTER
                        filter out instances based on a comma separated list
                        of feature_name=value,                             eg:
                        -i D#N_EDU=True,D#C_EDU=True
  -s, --simple          simplified attachment framework: every CDU features
                        are replaced by their head edu's features
  -r, --relations       add the annotated relation if any for each DU pair
                        (def: false)

============================================================
TODO:
  - extraction de traits syntaxiques ? tête du segment
 x- some features are wrong for cdu: need mapping cdu to edu members in document
   for timex, anaphora, simlex
 x- improved simlex for two sentences
 - improved timex upstream
 - should allow for one file for whole corpus, and use pointers to original files
   for feature extraction.
 - TODO: FeatureMap API changes:
         - add instance + rewrite constructor
         - generic feature map + subclass for annodis
 x- allow indexing based on set of features (eg: CandNode,NextNode).
   (only if one-to-one!)
 - debinarise features with given common prefix(es)
 - cache features: right now, same EDU gets processed several times
 x- input format parameters: sep for feature, sep for feat/value
 x- filtering of attributes
 x- filtering of instances

 - projection CDU-> (EDU head) looks suspicious (lotsa new danglers)
   eg
../../../data/attachment/train/Extr_frwikipedia_465_SermentsStrasbourg.features
Warning: no attachment point for EDU 1! Skipping.
Warning: no attachment point for EDU 6! Skipping.


  x- collapsing of features values, eg 4 classes covering 20 classes
  - collapsing of features (takes a set of features and a composition function, eg a logical or on a set of binary features)


"""

import sys
import os
import glob
import os.path
import codecs
import re
from collections import defaultdict
import xml.etree.ElementTree as ET
from AnnodisReader import annodisAnnot, Preprocess
try:
    from use_sqlite import init_voisins, get_voisins_list, get_voisins_dict
    LEX = True
except:
    print >> sys.stderr, "no usesqlite module, lexical neighbours ignored"
    LEX = False


from lexsim import set_similarity

DEF_ENCODING = "utf-8"

ITEM_SEP = None
VALUE_SEP = "="

DEBUG = False

# instances are indexed by two node reference, INDEX1 and INDEX2
INDEX1 = "SOURCE"
INDEX2 = "TARGET"
# used to be CandNode, NextNode, which was confusing as hell. 


class FeatureMap:
    """Stores set of instances, each instance holds set of (feature,value) pairs

    conventions for feature types, use prefix#name:
       - C#feat1: continuous feature
       - D#feat2: discrete
       - m#...  meta info
       - c#     target class
       - other?
    """

    def __init__(self, filename, item_sep = None, value_sep = "=", encoding = DEF_ENCODING, weird = False, empty = False):
        """reads feature map, one per line

        if weird is true, class attribute has lost its name and
        is located at position 8
        """
        if empty:
            return
        stuff = []
        domain = defaultdict(list)
        names = []
        in_file = codecs.open(filename, encoding = encoding)
        for line in in_file:
            feats = line.strip().split(item_sep)
            if weird:
                feats.insert(8, "CLASS")
            if value_sep is None:
                featmap = zip(feats[::2], feats[1::2])
            else:
                featmap = [x.split(value_sep) for x in feats]
            #print featmap
            #sys.exit()
            # featmap=[]
            # while feats!=[]:
            #     val=feats.pop()
            #     name=feats.pop()
            #     featmap.append((name,val))
            try:
                featmap = dict(featmap)
            except:
                print >> sys.stderr, featmap
                featmap = dict(featmap)
                sys.exit(0)
            stuff.append(featmap)
            names.extend([x for x in featmap.keys()])# if not(x.startswith("m#"))])
            for one in featmap:
                domain[one].append(featmap[one])
        in_file.close()
        self._all = stuff
        names = set(names)
        self._names = names
        for one in domain:
            domain[one] = set(domain[one])
        self._domain = domain
        self._index = {}

    def instances(self):
        return self._all

    def domain(self):
        return self._domain

    def names(self):
        return self._names


    def index(self, feat_list):
        """index instances on feature list, provided the list correspond to a unique instance,
        eg: (CandNode,NextNode) for attachment
        or (CandNode,NextNode,FILE)

        """
        self._index = {}
        self._indextype = feat_list
        previous = None
        for (i, one) in enumerate(self.instances()):
            vals = tuple([one.get(feat) for feat in feat_list])
            # if multiple instances with same id, make sure only the one classed
            # as True is indexed, as it is sure to be the right one. 
            # if all are classed as false, it does not matter
            if previous == vals and one.get("c#CLASS") == "False":
                pass
            else:
                self._index[vals] = i
                previous = vals


    def get_instance(self, instance_key):
        try:
            return self._all[self._index[instance_key]]
        except:
            print >> sys.stderr, "ERROR: out of bound instance index for :", instance_key, self._index.get(instance_key)
            print >> sys.stderr, "returning empty instance"
            sys.exit(0)#return {}


    def get_index(self, instance_key):
        return self._index[instance_key]

    def binarise(self):
        """non-boolean discrete attributes are split in boolean attributes

        ex: feat with value 1,2,3 will yield 3 boolean features feat_1,feat_2,feat_3
        """
        targets = []
        new = []
        for one in self.domain():
            if one.startswith("D#") and (len(self.domain()[one]) > 2 or (set() == self.domain()[one] & set(['True', 'False']))):
                targets.append(one)
                #print >> sys.stderr, one
                newfeats = [(x, "%s_%s" % (one, x)) for x in self.domain()[one]]
                for aninst in self.instances():
                    if aninst.has_key(one):
                        aninst.update(dict([(z, y == aninst[one]) for (y, z) in newfeats]))
                        del aninst[one]
                for (x, feat) in newfeats:
                    new.append(feat)
                self._names.remove(one)

        for one in targets:
            del self.domain()[one]
        for one in new:
            self.domain()[one] = set([True, False])
            self._names.add(one)



    def dump(self, filename, item_sep = " ", value_sep = "=", encoding = DEF_ENCODING, format = "sparse"):
        """dump the feature map with optional transformation or different formats

        possible format:
          - sparse: same as input (list feature,value pairs if feature is instantiated)
          - basket: orange sparse format (same as sparse but outputs 'feature=value')
          - full  : csv full feature instantiation
        """
        out = codecs.open(filename, "w", encoding = encoding)
        if format == "full":
            print >> out, ",".join(self._names)
        for one in self._all:
            if format == "sparse":
                print >> out, item_sep.join(["%s%s%s" % (x, value_sep, y) for (x, y) in one.items()])
            elif format == "basket":
                print >> out, " ".join(["%s=%s" % (x, y) for (x, y) in one.items() if not(y == "False")])
            else:
                this_instance = []
                for label in self._names:
                    value = one.get(label, "False")
                    if value == "False" and label.startswith("C"):
                        # continuous feature with missing value
                        value = "?"
                    this_instance.append(value)

                print >> out, ",".join(["%s" % x for x in this_instance])
        out.close()

    def dump_safe(self, filename, item_sep = " ", value_sep = "=", encoding = DEF_ENCODING, format = "sparse"):
        """dump the feature map with optional transformation or different formats

        possible format:
          - sparse: same as input (list feature,value pairs if feature is instantiated)
          - basket: orange sparse format (same as sparse but outputs 'feature=value')
          - full  : csv full feature instantiation
          - sanity: output the reference annotation, for sanity check

        SAFER: iterate on the index (THERE MUST BE ONE), to ensure unicity of instances
        DOES NOT CHECK THE 'True' INSTANCE IS PICKED -> this is taken care of at the index level

        """
        out = codecs.open(filename, "w", encoding = encoding)
        names = sorted(list(self._names))
        names.reverse()
        if format == "full":
            print >> out, ",".join(names)
        for idx in self._index:
            one = self.get_instance(idx)
            if format == "sparse":
                features = one.items()
                print >> out, item_sep.join(["%s%s%s" % (x, value_sep, y) for (x, y) in features])
            elif format == "basket":
                features = one.items()
                print >> out, " ".join(["%s=%s" % (x, y) for (x, y) in features if not(y == "False")])
            elif format == "sanity":
                if one.get("c#CLASS") == "True":
                    features = one
                    print >> out, item_sep.join([features["m#%s" % INDEX1], features["m#%s" % INDEX2], features["c#CLASS"]])
            else:
                this_instance = []
                for label in names:
                    value = one.get(label, "False")
                    if value == "False" and label.startswith("C"):
                        # continuous feature with missing value
                        value = "?"
                    this_instance.append(value)

                print >> out, ",".join(["%s" % x for x in this_instance])
        out.close()

    def process(self, doc, instance_func, propagate = False, strand_orphans = False):
        """enrich feature map by applying func on each instance

        instance_func must take as input the ids of candidate and next node
        and returns a dictionary of new features (at least one!)

        for cdu as cand_node, add features of head+tail segment wrt to nextnode
        for cdu as next_node add features of head only
        WARNING: specific to discourse features

        propagate replace feature for cdu by feature of its heads
        TODO: should be a different operation, happening after this ?
        """
        to_remove = set([])
        for (i, one) in enumerate(self._all):
            #print >> sys.stderr, "doing:", one["m#CandNode"],one["m#NextNode"]
            id1, id2 = one.get("m#%s" % INDEX1), one.get("m#%s" % INDEX2)
            cdu = False
            # to store head and tail EDU for complex segments
            cand_tail = None
            cand_head = None
            next_head = None
            if id1 is not None and id2 is not None:
                # complex segments features are propagated from their head/tail node
                # if propagate is true, complex segments is completely replaced by its head
                if one.get("D#C_EDU") == "False":
                    cand_head = one.get("m#C_RECURSIVE_HEAD")
                    cand_tail = one.get("m#C_RECURSIVE_TAIL")
                    cdu = True
                    id1 = cand_head
                    #print >> sys.stderr, "Cand Head =", id1
                    if id1 is None:# must be 0
                        id1 = one.get("m#%s" % INDEX1)

                if one.get("D#N_EDU") == "False":
                    next_head = one.get("m#N_RECURSIVE_HEAD")
                    cdu = True
                    #if propagate:
                    id2 = next_head
                    #print >> sys.stderr, "Next Head =", id2
                    if id2 is None:# must be 0
                        id2 = one.get("m#%s" % INDEX2)

                # if propagate, replace each cdu with its ALL its head features 
                # cdu-only features will be filtered out at a later stage
                # NB: would be simpler to change the class of (head,candidate), but then
                # we'd lose the trace of the operation
                catch = False
                if DEBUG and id1 == "12" and id2 == "13":
                    catch = True
                    print >> sys.stderr, "->  instance ", id1, id2

                if cdu and propagate:
                    if not(id1 == id2):
                        #print >> sys.stderr, "-> new instance ",id1, id2 
                        realinstance = self.get_instance((id1, id2))
                        real_inst_index = self.get_index((id1, id2))
                        to_remove.add(real_inst_index)
                        for each in realinstance:
                            if not(each.startswith("m#")) and not(each in ["c#CLASS"]):#,"D#N_EDU","D#C_EDU"]):
                                one[each] = realinstance[each]
                        if catch:
                            print one["c#CLASS"], realinstance["c#CLASS"]
                        if realinstance["c#CLASS"] == "True":
                            one["c#CLASS"] = "True"
                        else:
                            realinstance["c#CLASS"] = one["c#CLASS"]
                        # replace ids in the instances too ...
                        one["m#%s" % INDEX2] = id2
                        one["m#%s" % INDEX1] = id1
                        if catch:
                            print one["c#CLASS"], realinstance["c#CLASS"]
                    else:# next and cand have same head, don't try to attach an EDU to itself !
                        to_remove.add(i)

                if cdu and strand_orphans:
                    # an orphan is a lone node, attached to the EDU it's heading 
                    # so at this point, id1=id2
                    # -> detach it by removing the positive instance. 
                    if id1 == id2 and one["c#CLASS"] == "True":
                        to_remove.add(i)

                one.update(instance_func(doc, id1, id2))
                # add feature of the tail if candnode is a CDU 
                if cand_tail and not(propagate):
                    tailfeats = instance_func(doc, cand_tail, id2)
                    tailfeats = dict([(x + "_TAIL", y) for (x, y) in tailfeats.items()])
                    one.update(tailfeats)
                #print >> sys.stderr, "done:", one["m#CandNode"],one["m#NextNode"]
            else:
                print >> sys.stderr, "warning: instance has no candnode or nextnode", id1, id2

        self._all = [x for (i, x) in enumerate(self._all) if not(i in to_remove)]
        self.index(["m#%s" % INDEX1, "m#%s" % INDEX2])



    def filter_feats(self, feature_list, mode = "keep", continuous = True):
        """filter features; keep mode keeps only them, otherwise filter them out
        continuous = always keep continuous features no matter what
        """
        if mode == "keep":
            del_feature_list = set(self.names()) - set(feature_list)
        else:
            del_feature_list = set(feature_list)

        if continuous:
            del_feature_list = filter(lambda x: not(x.startswith("C")), del_feature_list)


        for feature in del_feature_list:
            if feature[0] not in ["c", "m"]:
                for one in self.instances():
                    if one.has_key(feature):
                        del one[feature]

        for one in del_feature_list:
            if one[0] not in ["c", "m"]:
                if one in self.domain():
                    del self.domain()[one]
                    self._names.remove(one)


    def filter_instances(self, condition_list):
        """ filter instances based on conditions, which are
        (for now) a feature name and a value

        condition must be met to keep the instance

        TODO: more general if condition are predicate functions, but less clumsy that way
        maybe two functions ?
        """
        bigcondition = lambda x: all([(x[f] == v) for (f, v) in condition_list])
        self._all = filter(bigcondition, self.instances())
        if self._index != {}:
            self.index(self._indextype)


    def discretize(self, featname, bins):
        """ discretize numerical feature featname, using bins as categories
        bins defines a list of contiguous intervals starting at first value
        eg [0,2,10,100] codes [-inf,0],[0-1],[2-10],[11-100],[100-inf]

        discrete values coded with feature first letter and sup value index
        """
        code = featname[0]#prefix category
        for one in self._all:
            val = int(one[featname])
            j = 0
            nb_bins = len(bins)
            while val > bins[j] and j < nb_bins - 1:
                j = j + 1
            one[featname] = "%s%d" % (code, bins[j])


    def extend(self, other):
        """add instances of another featuremap with self,
        and update feature names

        TODO: better merge of domains ...
        """
        newnames = []
        for one in other.instances():
            newnames.extend([x for x in one.keys()])
            self.instances().append(one)

        self._names |= set(newnames)


    def init_from_dir(self, path, suffix = "features"):
        files = glob.glob(os.path.join(path, "*" + suffix))
        print >> sys.stderr, files
        self.__init__(files[0])
        print >> sys.stderr, "init with file", files[0]
        for other_file in files[1:]:
            print >> sys.stderr, "adding file", other_file
            other = FeatureMap(other_file)
            self.extend(other)


    def collapse_values(self, featurename, new_classes):
        """use a set of super-classes to replace a more profligate set of classes
        new_classes is a dict of set
        """
        mapping = set2mapping(new_classes)
        for one in self._all:
            # if unknwon label, keep as is
            one[featurename] = mapping.get(one[featurename], one[featurename])





def set2mapping(sets):
    """ invert a dictionary of key:set to make a mapping from elements of sets to key
    eg   {"c1":set(1,2,3),"c2":set(4,5,6)} -> {1:"c1",2:"c1", ...}
    """
    result = {}
    for key in sets:
        for element in sets[key]:
            result[element] = key
    return result




def add_timex(doc, id1, id2):
    result = {}
    edu1 = doc.get_edu(id1)
    if edu1 is not None:
        result.update({"D#C_timex":(doc.get_edu(id1).attrib.get("timex", "False")),
                       "D#C_timex_signal":(doc.get_edu(id1).attrib.get("timex_signal", "None")),
                       "D#C_verb_class":(doc.get_edu(id1).attrib.get("verb_class", "unknown"))
                       })

    elif id1 == "0":
        pass
    else:
        print >> sys.stderr, "Warning Cand EDU %s not found in %s, probable CDU as HEAD of other CDU" % (id1, doc.name().decode("utf8"))
    edu2 = doc.get_edu(id2)
    if edu2 is not None:
        result.update({"D#N_timex":(doc.get_edu(id2).attrib.get("timex", "False")),
                       "D#N_timex_signal":(doc.get_edu(id2).attrib.get("timex_signal", "None")),
                       "D#N_verb_class":(doc.get_edu(id2).attrib.get("verb_class", "unknown"))})
    elif id1 == "0":
        pass
    else:
        print >> sys.stderr, "Warning NextNode EDU %s not found in %s, probable CDU as HEAD of other CDU" % (id2, doc.name().decode("utf8"))
    return result


def add_verbclass(doc, id1, id2):
    result = {}
    edu1 = doc.get_edu(id1)
    if edu1 is not None:
        result.update({"D#C_verb_class":(doc.get_edu(id1).attrib.get("verb_class", "unknown"))
                       })
    else:
        if not(id1 == "0"):
            print >> sys.stderr, "Warning EDU %s not found in %s, probable CDU as HEAD of other CDU" % (id1, doc.name().decode("utf8"))
    edu2 = doc.get_edu(id2)
    if edu2 is not None:
        result.update({"D#N_verb_class":(doc.get_edu(id2).attrib.get("verb_class", "unknown"))})
    else:
        if not(id2 == "0"):
            print >> sys.stderr, "Warning EDU %s not found in %s, probable CDU as HEAD of other CDU" % (id2, doc.name().decode("utf8"))
    return result


def add_relations(doc, id1, id2):
    """for pair of DU, add the type of relation between if there is one, None otherwise
    """

    result = {}
    result["D#relation"] = doc.get_rel(id1, id2)
    return result




# superseded by lexsim module
# TODO: different lexical similarity measures
#    x - mihalcea: ~ average of max similarity pairs
#      - sum of pairs normalised by number of pairs (used for topic segmentation, and probably very crappy)
#      - max similarity pair
#      - number of repetitions normalised by sentence lengths
#      - exclude repetition or not
#      - variants with part-of-speech: n,v,adj
#      - syntax-aware similarity: focus on V-V, V-obj, attribute ADJ ?
#        other ? check with clementine


# def set_similarity(set1,set2,sim_func,mode="mihalcea",repetition=True):
# 	"""
# 	compute similarity between two sets 
# 	based on a similarity function between elements
# 	mode can be :
# 	todo- max:  max(sim(x,y) for (x,y) in (S1 X S2))
# 	- mihaleca: 1/2(sum_w1 max sim w1/S2 + sum_w2 max sim w2/S1)
# 	? todo- hausdorff equivalent: 1-hausdorff distance 
# 	 hausdorff distance = max (max_S1 dist(x,S2), max_S2 dist(y,S1)) 

# 	- repetition: whether to take repetitions into account or not 

# 	returns a real value and a trace of the computation:
# 	for max, it is the max pair
# 	for mihalcea the sets of max pairings
# 	for hausdorff the max pairs used for the computation
# 	"""

#        	total1=0
#     	nb_paires = 0
# 	trace = []
# 	if list(set1) == [] or list(set2)==[]:
# 		return 0 
#     	for one in set2:
# 		closest = max(map(lambda x:sim_func(one,x),set1))
#             	total1 += closest
#             	nb_paires += 1
#    	if nb_paires > 0:
#         	total1 = total1 / nb_paires
#    	nb_paires = 0
#    	total2 = 0
#    	for one in set1:
#         	closest = max(map(lambda x:sim_func(one,x),set2))
#             	total2 += closest
#             	nb_paires += 1
#     	if nb_paires > 0:
#         	total2 = total2 / nb_paires
#     	total = total1 + total2 
#     	total = total / 2.0
# 	return total	
#     #print total


def edu_similarity(doc, id1, id2, restriction = ["N", "V"], stop_words = set()):
    result = {}
    edu1_toks = set([x.lemma() for x in doc.get_edu_tokens(id1) if x.simple_mp_cat() in restriction]) - stop_words
    edu2_toks = set([x.lemma() for x in doc.get_edu_tokens(id2) if x.simple_mp_cat() in restriction]) - stop_words
    sim_func = lambda x, y: (doc._voisins.get(x, {})).get(y, 0)
    result["C#SIMLEX"] = set_similarity(edu1_toks, edu2_toks, sim_func, mode = "mihalcea", debug = False)#weight=lambda x:1.0)
    return result



def add_voisins(doc, id1, id2, details = False):
    """sum of lin similarity of lexical pairs that are similar in two EDUs

    if details is True, records best pairs of (w1 in id1,w2 in id2)

    TODO:
       - bugged as hell: redo from scratch (above, with refactroring)
       x- better to get all lexical neighbours first for the whole text and
       then do the extraction
       - still better to structure voisins as a map indexed on both lemmas, rather than adj list ?
       x- lemmatise
       x- restrict N/V
       - take max sim for all same lemma-pairs
       x- better measure ... (normalised or restricted to verbs/verbs for instance ?)
         test with Mihalcea: 1/2(sum_w1 max sim w1/S2 + sum_w2 max sim w2/S1)
         sans idf (comment ?)
       x- complex segments ? (=head+tail, done elsewhere)
    """
    result = {}
    edu1 = doc.get_edu(id1)
    if edu1 is not None:
        edu1_toks = [x.lemma() for x in doc.get_edu_tokens(id1) if x.simple_mp_cat() in ["N", "V"]]
        #print edu1_toks
        edu1_vsn = {}
        for one in edu1_toks:
            edu1_vsn.update(dict(doc._voisins[one]))
    else:
        edu1_toks = []
        edu1_vsn = {}

    edu2 = doc.get_edu(id2)
    if edu2 is not None:
        edu2_toks = [x.lemma() for x in doc.get_edu_tokens(id2)]
        edu2_vsn = {}
        for one in edu2_toks:
            edu2_vsn.update(dict(doc._voisins[one]))

    else:
        edu2_toks = []
    # all pairs
    total = 0
    nb_paires = 0
    for one in edu2_toks:
        closest = 0
        if one in edu1_vsn:
            closest = max(closest, edu1_vsn[one])
            total += closest
            nb_paires += 1
    if nb_paires > 0:
        total = total / nb_paires
    nb_paires = 0
    total2 = 0
    for one in edu1_toks:
        closest = 0
        if one in edu2_vsn:
            closest = max(closest, edu2_vsn[one])
            total2 += closest
            nb_paires += 1
    if nb_paires > 0:
        total2 = total2 / nb_paires
    total = total + total2
    total = total / 2.0

    #print total
    result["C#SIMLEX"] = total
    return result


# pers_pro = re.compile("")  
# poss_pro =
# def_art = 
# indep_art =
# pn = 
# dem_art / dem_pro


def add_anaph_feats(doc, id1, id2, which = "pairwise"):
    """add anaphora features:

    which : which set of features to consider:
           - edu1: features specific to first edu
           - edu2: ...
           - pairwise: relational features between the 2 edus
           - all: all of them
    """


    if id1 == id2:
        print >> sys.stderr, "Error: EDU %s <==> EDU %s!" % (id1, id2)

    result = {}
    # mention functions
    mention_fcns = ["is_pro", "is_il", "is_cl_pro", "is_non_refl_cl_pro", \
                    "is_name", "is_short_name", \
                    "is_def_np", "is_short_def_np", \
                    "is_dem_np", "is_short_dem_np", \
                    "is_indef_np", \
                    "is_poss_np"]
    # matching functions
    match_fcns = ["string_match", "head_match", "pro_match"\
                  "gender_agree", "number_agree", "morph_agree"]


    # Mention features in EDU1
    mentions1 = []
    edu1 = doc.get_edu(id1)
    if edu1 is not None: # candidate node
        mentions1 = doc.get_edu_mentions(id1)
        # print edu1, mentions1
        # Mention features in EDU1
        edu1_counts = {}
        for m1 in mentions1:
            # print m1
            for fcn in mention_fcns:
                fcn_res = getattr(m1, fcn)()
                # print "  >>", fcn, fcn_res
                if which == "all" or which == "edu1":
                    result.update({"D#EDU1-" + fcn: fcn_res})
                if fcn_res == True:
                    edu1_counts[fcn] = edu1_counts.get(fcn, 0) + 1
        # print doc.get_edu_text(id1).encode('utf8'), edu1_counts
        if which == "all" or which == "edu1":
            for mtype, ct in edu1_counts.items():
                result.update({"C#EDU1-" + mtype: ct})
    else:
        if id1 != "0":
            print >> sys.stderr, "Warning EDU %s not found in %s, probable CDU as HEAD of other CDU" % (id1, doc.name().decode("utf8"))


    #
    # Mention features in EDU2
    mentions2 = []
    edu2 = doc.get_edu(id2)
    if edu2 is not None: # EDU to attach
        mentions2 = doc.get_edu_mentions(id2)
        edu2_counts = {}
        for m2 in mentions2:
            # print m2
            for fcn in mention_fcns:
                fcn_res = getattr(m2, fcn)()
                # print "  >>", fcn, fcn_res
                if which == "all" or which == "edu2":
                    result.update({"D#EDU2-" + fcn:fcn_res})
                if fcn_res == True:
                    edu2_counts[fcn] = edu2_counts.get(fcn, 0) + 1
        if which == "all" or which == "edu2":
            for mtype, ct in edu2_counts.items():
                result.update({"C#EDU2-" + mtype: ct})
    else:
        if id2 != "0":
            print >> sys.stderr, "Warning EDU %s not found in %s, probable CDU as HEAD of other CDU" % (id2, doc.name().decode("utf8"))

    # pairwise features
    if which == "pairwise":
        for m2 in mentions2: # ana
            for m1 in mentions1: # ante
                #print m1, m2, m1.name_match(m2), m1.pro_match(m2), m2.compat_pro_ante(m1)
                result.update({"D#name_match": m1.name_match(m2)})
                result.update({"D#pro_match": m1.pro_match(m2)})
                result.update({"D#pro_compat_ante": m1.compat_pro_ante(m2)})

    # TODO: features that combine ling form and distance
    return result


def add_nothing(doc, id1, id2):
    """vacuous update of features, used to propagate cdu heads even when no other feature
    addition is asked for
    """
    return {}


# two-class classification of relations / subord-coord
_class_schemes = {
    "subord_coord":{
        "subord":set(["elaboration", "e-elab", "attribution", "comment", "flashback", "explanation", "alternation"]),
        "coord": set(["continuation", "parallel", "contrast", "temploc", "frame", "narration", "conditional", "result", "goal", "background"]),
        "NONE":set(["null", "unknown", "NONE"])
        },
    # four class +/- closer to PDTB
    "pdtb": {
        "contingency": set(["explanation", "conditional", "result", "goal", "background"]),
        "temporal": set(["temploc", "narration", "flashback"]),
        "comparison":set(["parallel", "contrast"]),
        "expansion":set(["frame", "elaboration", "e-elab", "attribution", "continuation", "comment", "alternation"]),
        "error":set(["null", "unknown", "NONE"])
        },
    # our own hierarchy
    "minsdrt":{
        "structural": set(["parallel", "contrast", "alternation", "conditional"]),
        "sequence": set(["result", "narration", "continuation"]),
        "expansion": set(["frame", "elaboration", "e-elab", "attribution", "comment", "explanation"]),
        "temporal": set(["temploc", "goal", "flashback", "background"]),
        "error":set(["null", "unknown", "NONE"])
        }
}

# coord -> contingency+structural, subord -> expansion + temporal



if __name__ == "__main__":
    import optparse

    usage = "usage: %prog [options] file basename"
    parser = optparse.OptionParser(usage = usage)
    parser.add_option("-t", "--timex", default = False, action = "store_true",
                      help = "add timex (def: False)")
    parser.add_option("-c", "--verb-class", default = False, action = "store_true",
                      help = "add event verb class (def: False)")
    parser.add_option("-a", "--anaphor", default = False, action = "store_true",
                      help = "add anaphor (def: False)")
    parser.add_option("-v", "--voisins", default = None, action = "store",
                      help = "lex neighbors DB ")
    parser.add_option("-w", "--weird", default = False, action = "store_true",
                      help = "if input is in weird format for class attribute")
    parser.add_option("-d", "--distance", default = False, action = "store_true",
                      help = "discretize distance")
    parser.add_option("-f", "--format", default = "sparse", type = "choice", choices = ["sparse", "full", "basket", "sanity"],
                      help = "output format (sparse of full); sanity outputs the reference, just for verification")
    parser.add_option("-o", "--output", default = None,
                      help = "output destination (default: inplace modification)")
    parser.add_option("-m", "--merge", default = None,
                      help = "merge all features files with given suffix into one big output; \
                           do not try to add features at the same time (undefined results will bite you in the )!")
    parser.add_option("-b", "--binarise", default = False, action = "store_true",
                      help = "binarise discrete non boolean features")
    parser.add_option("-l", "--filter", default = None, action = "store",
                      help = "use given file to filter out features")
    parser.add_option("-k", "--kill", default = False, action = "store_true",
                      help = "kill selected features instead of keeping them (def: false)")
    parser.add_option("-i", "--inst-filter", default = None, action = "store",
                      help = "filter out instances based on a comma separated list of feature_name=value, \
                            eg: -i D#N_EDU=True,D#C_EDU=True")
    parser.add_option("-s", "--simple", default = False, action = "store_true",
                      help = "simplified attachment framework: every CDU features are replaced by their head edu's features")
    parser.add_option("-r", "--relations", default = False, action = "store_true",
                      help = "add the annotated relation if any for each DU pair (def: false)")

    parser.add_option("-p", "--strand-orphans", default = False, action = "store_true",
                      help = "remove attachments of orphan nodes to their CDUs")
    parser.add_option("-y", "--class-scheme", default = "sdrt", type = "choice", choices = ["sdrt", "pdtb", "minsdrt", "subord_coord"],
                      help = "merge classes according to given relation grouping (cf doc)")

    (options, args) = parser.parse_args()




    allfuncs = {}
    if options.timex:
        allfuncs["timex"] = add_timex
    if options.anaphor:
        allfuncs["anaph"] = add_anaph_feats
    if options.verb_class:
        allfuncs["verb"] = add_verbclass
    if options.voisins and LEX:
        #allfuncs["voisins"] = add_voisins
        db_filename, table = init_voisins(options.voisins)
        allfuncs["voisins"] = edu_similarity
    if options.relations:
        allfuncs["relations"] = add_relations

    if (options.simple or options.strand_orphans) and allfuncs == {}:
        allfuncs["dummy"] = add_nothing

    print >> sys.stderr, "Requested additions:", allfuncs.keys()

    # basename = args[0]
    # doc      = annodisAnnot(basename+".xml")
    # feats    = FeatureMap(basename+".features")
    # prep     = Preprocess( basename+".txt.prep.xml")

    if options.merge:
        feats = FeatureMap("", empty = True)
        feats.init_from_dir(".", suffix = options.merge)
        basename = "no base file to consider"
    else:
        if True:
            basename = args[0]
            feats = FeatureMap(basename + ".features", weird = options.weird)
            feats.index(["m#%s" % INDEX1, "m#%s" % INDEX2])
            if allfuncs != {}:
                doc = annodisAnnot(basename + ".xml")
                prep = Preprocess(basename + ".txt.prep.xml")
                doc.add_preprocess(prep)
            if options.voisins:
                doc._voisins = get_voisins_dict(table, doc._vocab)
                for entry in doc._voisins:
                    doc._voisins[entry] = dict(doc._voisins[entry])
        else:
            print >> sys.stderr, "Usage: script file-basename ?", args
            sys.exit(0)

    if options.merge:
        feats.index(["m#%s" % INDEX1, "m#%s" % INDEX2, "m#FILE"])
    else:
        feats.index(["m#%s" % INDEX1, "m#%s" % INDEX2])

    for onename, onefunc in allfuncs.items():
        feats.process(doc, onefunc, propagate = options.simple, strand_orphans = options.strand_orphans)
        print >> sys.stderr, onename, " done"
    if options.distance:
        try:
            feats.discretize("DISTANCE", [0, 1, 4, 10])
        except KeyError:
            print >> sys.stderr, "no DISTANCE feature to discretize!"
    if options.binarise:
        feats.binarise()

    if options.class_scheme != "sdrt":
        feats.collapse_values("c#CLASS", _class_schemes[options.class_scheme])


    # first filter instances
    if options.inst_filter:
        try:
            conditions = [x.split("=") for x in options.inst_filter.split(",")]
        except:
            print >> sys.stderr, "ERROR: wrong format for conditions", options.inst_filter
            sys.exit(0)
        feats.filter_instances(conditions)
    # then maybe features
    if options.filter:
        feature_list = set([x.strip() for x in codecs.open(options.filter).readlines() if x.strip() != ""])
        feats.filter_feats(feature_list, mode = ["keep", "lose"][options.kill], continuous = False)

    print >> sys.stderr, basename, " done"

    if options.output:
        output = options.output
    else:
        output = basename + ".features"

    #if len(feats._index.keys())!=len(feats._all):
    #    print >> sys.stderr, "WARNING: duplicate instances in ", basename


    feats.dump_safe(output, format = options.format)
