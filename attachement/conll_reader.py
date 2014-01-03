#!/usr/bin/env python

import codecs


class ConllReader( object ):

    """Data reader for Conll

    ID      FORM    LEMMA   CPOSTAG POSTAG  FEATS   HEAD    DEPREL  PHEAD   PDEPREL
    1       Il      cln     CL      CL      -       4       suj     -       -
    2       ne      ne      ADV     ADV     -       4       mod     -       -
    3       s'      se      CL      CL      -       4       aff     -       -
    4       agit    agir    V       V       -       0       root    -       -
    5       pas     pas     ADV     ADV     -       4       mod     -       -
    ...
    
    Reader is an iterator outputting lists (of word tuples)
    representing sentences

    """

    def __init__(self,infile,encoding="utf-8"):
        self.stream = codecs.open(infile,'r',encoding)
        return

    def __iter__(self):
        return self 

    def next(self):
        items = []
        while True:
            line = self.stream.readline()
            # end of file
            if not line:
                if items != []:
                    return items # a sentence
                self.stream.seek(0)
                raise StopIteration
            # end of block
            elif line and not line.strip():
                if items != []:
                    # print items
                    return items
            # content line
            else: # accumulate token: i.e., list
                items.append( line.strip().split() )






def test( filename, ln=10 ):
    err_ct = 0
    data = ConllReader( filename )
    l_ct = 0
    for sentence in data:
        for token in sentence:
            # print tok
            l_ct += 1
            if len(token) != ln:
                err_ct += 1
                print "Error in line %s: '%s'" %(l_ct," ".join(token)) 
        l_ct += 1
    print "Number of errors:", err_ct
    return


                
        
if __name__ == "__main__":
    import sys
    test( sys.argv[1] )

