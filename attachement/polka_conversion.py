#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
read *.features file and convert them into POLKA formats:
classification or ranking.
"""

import sys
import os
import codecs
# import ast
from add_feature import FeatureMap


POLKA_SEP=":"
ANA_NODE="m#NextNode"
ANTE_NODE="m#CandNode"
CLASS_ATTR="c#CLASS"



def _esc_sep(feat,sep=POLKA_SEP):
    esc_sym = u"*"
    return feat.replace(sep,esc_sym)


def _binarize(fname,fvalue,sep=POLKA_SEP):
    """ return binarized features """
    fname = _esc_sep("%s=%s" %(fname,fvalue))
    fvalue = 1.0
    return u"%s%s%s" %(fname,sep,fvalue)


def _real_format(fname,fvalue,sep=POLKA_SEP):
    """ format feature-value pair so that each value is a float """
    if fname.startswith("C#"):
        fvalue = float(fvalue)
    # try:
    #     # fvalue = ast.literal_eval(fvalue)
    #     fvalue = eval(fvalue)
    # except ValueError:
    #     pass
    # if isinstance(fvalue,float):
    #     pass
    # elif isinstance(fvalue,bool) or isinstance(fvalue,int):
    #     fvalue = float(fvalue)
    else: # binarize other feature (e.g., strings)
        fname = u"%s=%s" %(fname,fvalue)
        fvalue = 1.0
    fname = _esc_sep(fname)
    return u"%s%s%s" %(fname,sep,fvalue)


def feat_map2_class_instance( feat_map, formatting=_real_format, features2cross=[], features2filter=[]):
    cl = feat_map.pop( CLASS_ATTR )
    cl = int(eval(cl) == True)
    ante = feat_map.pop( ANTE_NODE )
    ana = feat_map.pop( ANA_NODE )
    # remove other meta info    
    feats = [(f,v) for (f,v) in feat_map.items()
             if not f.startswith("m#")] 
    # feature combinations # WARNING: only binary for now
    fdict = dict( feats )
    for f1 in features2cross:
        val1 = fdict[f1]
        fv1 = "%s=%s" %(f1,val1)
        for f2,v in fdict.items():
            # NB: f2 prefix needs to appear first
            feats.append( ("%s-%s" %(f2,fv1), v) )
    # feature conversion
    feats = [formatting(f,v) for (f,v) in feats]
    return (ana, ante), (cl, feats)




class PolkaConvertor( object ):
    
    def __init__( self, indir, suffix=".features", in_enc="utf-8", out_enc="utf-8", formatting=_real_format, features2cross=[], features2filter=[] ):
        self._dir = indir
        self._in_encoding = in_enc
        self._out_encoding = out_enc
        self._formatting = formatting
        self._files = [os.path.join(indir,f) for f in os.listdir(indir)
                       if f.endswith(suffix)]
        self._features2cross = features2cross
        self._features2filter = features2filter
        assert self._files != [], "No file!"
        print >> sys.stderr, "Nber of feature files:", len(self._files)
        return
    

    def classifier_instances(self):
        for f in self._files:
            print >> sys.stderr, f
            feat_map = FeatureMap(f, encoding=self._in_encoding)
            # feature filtering
            feat_map.filter_feats(self._features2filter,
                                  mode="lose",
                                  continuous=False) # WARNING: add hoc!!!
            for instance in feat_map._all:
                edu_pair, (cl, feats) = feat_map2_class_instance( instance,
                                                                  formatting=self._formatting,
                                                                  features2cross=self._features2cross )
                # print edu_pair[0], edu_pair[1], cl
                yield (edu_pair, f), cl, feats


    def dump_class_instances(self, filename=None):
        if filename:
            f = codecs.open(filename, "w", encoding=self._out_encoding)
        else:
            f = sys.stdout
        ct = 0
        for _, cl, feats in self.classifier_instances():
            ct += 1
            print >> f, "%s %s" %(cl," ".join(feats)) 
        f.close()
        print >> sys.stderr, "Nber of instances:", ct
        return 


    def ranker_instances(self):
        for f in self._files:
            print >> sys.stderr, f
            feat_map = FeatureMap(f, encoding=self._in_encoding)
            # feature filtering
            feat_map.filter_feats(self._features2filter,
                                  mode="lose",
                                  continuous=False) # WARNING: add hoc!!!
            edu_ante_list = {}
            # build attachment point correct/incorrect candidate lists 
            for instance in feat_map._all:
                (ana, cand), (cl, feats) = feat_map2_class_instance( instance,
                                                                     formatting=self._formatting,
                                                                     features2cross=self._features2cross )
                # print ana, cand, cl
                edu_ante_list[ana] = edu_ante_list.get(ana,[]) + [(cand,cl,feats)]
            # make sure data always get ordered the same way
            edu_ante_list = edu_ante_list.items()
            edu_ante_list.sort()
            # generate ranking instances
            for edu, cand_list in edu_ante_list:
                correct_cands = [(cand,cl,feats) for (cand,cl,feats) in cand_list
                                 if cl == 1]  
                if len(correct_cands) == 0:
                    print >> sys.stderr, "Warning: no attachment point for EDU %s! Skipping." %edu
                    #print >> sys.stderr, edu, (cand_list)
                #elif len(correct_cands) > 1:
                #    print >> sys.stderr, "Warning: multiple attachment points for EDU %s! Skipping." %edu
                else:
                    yield edu, cand_list, f
                    

    def dump_rank_instances(self, filename):
        if filename:
            f = codecs.open(filename, "w", encoding=self._out_encoding)
        else:
            f = sys.stdout
        ct = 0
        for _, cand_list, _ in self.ranker_instances():
            ct += 1
            print >> f, len(cand_list)
            for _,cl,feats in cand_list:
                print >> f, "%s %s" %(cl," ".join(feats)) 
        f.close()
        print >> sys.stderr, "Nber of instances:", ct
        return 



if __name__ == '__main__':
    import sys
    import os
    import optparse
    usage = "usage: %prog [options] dir"
    parser = optparse.OptionParser(usage=usage)
    parser.add_option("-f", "--format", choices=["class","rank"], default="class", help="polka format (default: 'class')")
    parser.add_option("-o", "--output_file", default=None, action="store", help="output file (def: stdout)")
    parser.add_option("-k", "--crossing", default="", action="store", help="features to cross with others (only binary for now) (default: '')")
    # feat dict
    (options, args) = parser.parse_args()

    feats2cross = []
    if options.crossing != "":
        feats2cross = options.crossing.split(",") # e.g., D#DIRECTIONALITY_LEFT, D#C_EDU
    
    in_dir = args[0]
    conv = PolkaConvertor( in_dir, features2cross=feats2cross )
    outfile = options.output_file

    # classifier format
    if options.format == "class":
        conv.dump_class_instances( outfile )
        

    # ranker format
    elif options.format == "rank":
        conv.dump_rank_instances( outfile )




