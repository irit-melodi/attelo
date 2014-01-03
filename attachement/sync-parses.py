#!/usr/bin/env python
# -*- coding: utf-8 -*-


import sys
import os
import codecs
import re
import xml.etree.ElementTree as ET


DEBUG=False 


"""
python sync-parses.py <dir containing .txt, .inmst, and .outmst files>

e.g., python sync-parses.py data/attachment/train/

"""


def normalize_token_form( form ):
    """ bunch of hacks to take into account MWEs and their (sometimes
    peculiar) preprocessing: addition of underscores and conversion of
    some characters like 'à' (lc-ing, de-accenting), etc.
    """
    form = form.replace(u"'_",u"'") # e.g., aujourd'_hui =>  aujourd'hui
    form = form.replace(u"_",u" ") # e.g., tout_au_long_de  =>  tout au long de
    form = normalize_as( form ) # e.g., à priori => a priori
    form = form.replace(u"au dessus",u"au-dessus")
    # form = form.replace(u"d'autre",u"d' autre") # only for WK_-_exobiologie.txt
    return form


def normalize_text( text ):
    """ deal with special characters in raw text, normalize the a's
    and other oddities. NOTE: none of these operations should change
    the character count."""
    text = text.replace(u" ",u" ")
    # text = text.replace(u"\r",u"") # WARNING: this actually alters char count
    text = normalize_as( text )
    text = text.replace(u"au dessus",u"au-dessus") 
    return text 


def normalize_as( form ): 
    form = form.replace(u"à",u"a")
    form = form.replace(u"À",u"a")
    form = form.replace(u"A",u"a")
    return form



def sync_document( mstout_data, mstin_data, raw_data ):
    ''' recover original offsets (in text data) for the tokens found
    in the parses

    raw_data: original text
    
    mstin: input to MST parser (lemmas, POS-tags, morph. features,
    clusters) in pseudo-CONLL format:

    ID LEMMA _ CPOSTAG POSTAG FEATS 0 d _ _ FORM

    mstout: output of MST parser (contains synt. deps but some info
    are gone: lemmas, morpho features, etc.) in original CONLL
    format:

    ID FORM LEMMA CPOSTAG POSTAG FEATS HEAD DEPREL PHEAD PDEPREL
    '''
    txt = raw_data.read().rstrip()
    txt_ln = len(txt)
    pos = 0
    sentences = []
    for parse in mstout_data:
        tokens = []
        pre_parse = mstin_data.next()
        for i,token in enumerate(parse):
            form = token[1]
            # normalize strings
            form = normalize_token_form( form )
            if DEBUG:
                print >> sys.stderr, "Token: '%s' <=> TXT: '%s'" %(form.encode('utf-8'), txt[:20].encode('utf-8'))
            # find offsets
            form_re = re.compile( re.escape(form), re.IGNORECASE ) 
            match = form_re.search( normalize_text(txt) )
            if not match:
                print >> sys.stderr, "ERROR: Regex '%s' not found in string: '%s'..." \
                      %(form.encode('utf-8'),txt[:20].encode('utf-8'))
                raise re.error
            start,end = match.span()
            start1 = start + pos
            end1 = end + pos
            if DEBUG:
                print >> sys.stderr, start, end, start1, end1
            # shift cursor
            pos += end
            txt = txt[end:]
            # update CONLL fields using info from .inmst file
            token1 = pre_parse[i]
            token[2] = token1[1] # lemma
            token[4] = token1[4] # POSTAG
            token[5] = token1[5] # FEATS
            # print form.encode('utf-8'), start1, end1
            token.append( str(start1) )
            token.append( str(end1) )
            tokens.append( token )
        sentences.append( tokens )
    # test
    if pos != txt_ln:
        print pos, txt_ln
        print >> sys.stderr, "Error: some of the original text has not been found in *mst files."
        sys.exit()
    return sentences



CONLL_ATTR_NAMES = ['ID','FORM','LEMMA','CPOSTAG','POSTAG',\
                    'FEATS','HEAD','DEPREL','PHEAD','PDEPREL']
EXTENT_ATTR_NAMES = ['START','END']


def build_xml( parses, src_filename ):
    doc = ET.Element('document', {'source': src_filename} )
    for i,parse in enumerate(parses):
        parse_elt = ET.SubElement(doc, 'mst_parse', {'id': str(i)} )
        for token in parse:
            tok_elt = ET.SubElement(parse_elt,'token')
            conll_attrs = dict(zip(CONLL_ATTR_NAMES,token[:10]))
            ext_attrs = dict(zip(EXTENT_ATTR_NAMES,token[10:]))
            conll = ET.SubElement(tok_elt,'conll', conll_attrs)
            extent = ET.SubElement(tok_elt,'extent', ext_attrs)
    xml_tree = ET.ElementTree(doc)
    return xml_tree




if __name__ == "__main__":
    import sys
    import codecs
    from conll_reader import ConllReader

    err_ct = 0
    # get the files
    indir = sys.argv[1]
    txt_files = dict([(f,os.path.join(indir, f)) for f in os.listdir(indir) if f.endswith(".txt")])
    inmst_files = dict([(f[:-6],os.path.join(indir, f)) for f in os.listdir(indir) if f.endswith(".inmst")])
    outmst_files = dict([(f[:-7],os.path.join(indir, f)) for f in os.listdir(indir) if f.endswith(".outmst")])
    # synchronize 'em
    print >> sys.stderr, "Sync-ing %s file(s)..." %len(txt_files)
    for f in txt_files:
        print >> sys.stderr, ">>>", f
        txt_file = txt_files.get(f)
        mstin_file = inmst_files.get(f)
        mstout_file = outmst_files.get(f)
        if not mstin_file or not mstout_file:
            print >> sys.stderr, "Warning: no parse for file %s" %f
        else:
            try:
                parses = sync_document( ConllReader(mstout_file),
                                        ConllReader(mstin_file),
                                        codecs.open(txt_file, 'r', 'utf-8') )
                filename = os.path.basename(txt_file)
                xml = build_xml( parses, filename )
                xml_file = os.path.join(indir, filename+".prep.xml")
                xml.write(xml_file, encoding="utf-8")
                os.system('xmllint --format %s > tmp; mv tmp %s' %(xml_file,xml_file))
            except re.error:
                print >> sys.stderr, "Could not sync file %s" %f
                err_ct += 1
    print >> sys.stderr, "\ndone."
    print >> sys.stderr, "%s errors out of %s files." %(err_ct,len(txt_files))
            

