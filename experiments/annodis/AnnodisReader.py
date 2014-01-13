#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Api for Annodis annotation in xml format

also handles preprocessed files with postag+dependency 


TODO: 
  x- api on EDUs (they are now just elementtree elements)
  x- api on CDUs (they are now just elementtree elements)
  x- corpus object
"""

try:
    import xml.etree.cElementTree as ET
except:
    import xml.etree.ElementTree as ET
import sys
import glob
import os.path
import random
import codecs

from collections import defaultdict
try:
    from use_sqlite import init_voisins, get_voisins_list, get_voisins_dict
    from make_sqlite import get_syno_norm_dict
    LEX=True
except:
    print >> sys.stderr, "no usesqlite module, lexical neighbours ignored"
    LEX=False


DEFAULT_ENCODING="utf-8"

# regexes
import re
FEM_PERS_PRO = re.compile(r'^elle(s)?$',re.IGNORECASE)
MASC_PERS_PRO = re.compile(r'^(il(s)?|lui|eux)$',re.IGNORECASE)
THIRD_SG_PERS_PRO = re.compile(r'^(elle|il|lui|l\')$',re.IGNORECASE)
THIRD_PL_PERS_PRO = re.compile(r'^(ils|elles|eux)$',re.IGNORECASE)
# ...


# should not be here
class Token( object ):
    """ 
    """
    def __init__(self,elt_xml,format="xml"):
        self._deps = [] # list of (direct) dependent token
        self._gov = None # governing token
        self._sentence = None
        if format=="xml":
            # feats=elt_xml.attrib
            feats = elt_xml.find("conll").attrib
            ext = elt_xml.find("extent")
            self._extent = int(ext.attrib["START"]),int(ext.attrib["END"])
            self._head = feats["HEAD"]
            self._drel  = feats["DEPREL"]
            self.p  = feats["CPOSTAG"] # un peu vicieux le nom des attributs: p et c?!
            self.c  = feats["POSTAG"]
            self.id = str(int(feats["ID"]))
            self.i  = str(int(feats["ID"])-1)# position
            self.f  = feats["FORM"] 
            self.l  = feats["LEMMA"]
            self.morpho=feats["FEATS"].split("|")
            if self.morpho == ["_"]:
                self.morpho = {}
            else:
                self.morpho=dict([x.split("=") for x in self.morpho])
                
        elif format=="conll":
            token=elt_xml.split("\t")
            self.c=token[4]# mp tag
            self.p=token[3]# mp coarse tag
            self.f=token[1]# form
            self.l=token[2]#lemma
            if self.l=="_":
                self.l=self.f
            self.i=str(int(token[0])-1)#position
            self.id=str(int(token[0]))
            self._head=str(int(token[6])-1)
            self._drel=token[7]
        elif format=="malt":
            if type(elt_xml)==type(""):
                token=elt_xml.split("\t")
            else:
                token=elt_xml
            self.c=token[4]# mp tag
            self.p=token[3]# mp coarse tag
            self.f=token[1]# form
            self.l=token[2]#lemma
            if token[5]=="_":
                self.morpho={}
            else:
                self.morpho=token[5].replace("|",",")
                self.morpho=dict([x.split("=") for x in self.morpho.split(",")])
            self.i=str(int(token[0])-1)#position
            self.id=str(int(token[0]))
            self._head=str(int(token[7])-1)
            self._drel=token[8]
        else:
            print >> sys.stderr, "unknown format:",format
            sys.exit(0)

    _legal_supertags=set("A Cc Cs D E N NP O P R T V".split())

        
    def __repr__(self,encoding=DEFAULT_ENCODING):
        return ("/".join((self.lemma(),self.mp_cat(),str(self.position())))).encode(encoding)

    def __str__(self,encoding=DEFAULT_ENCODING):
        return ("/".join((self.lemma(),self.mp_cat(),str(self.position())))).encode(encoding)

    def __le__(self,other):
        return self.position()<=other.position()

    def __lt__(self,other):
        return self.position()<other.position()

    def __eq__(self,other):
        return self.position()==other.position() and self.lemma==other.lemma

    def extent(self):
        assert isinstance(self._extent,tuple) 
        return self._extent
    
    def mp_cat(self):
        """morpho-syntactic category"""
        return self.c

    def simple_mp_cat(self):
        """coarser-grained morpho-syntactic category (supertags)"""
        return self.p

    def lemma(self):
        """lemma"""
        return self.l

    def word_form(self):
        return self.f.replace("_"," ").replace("' ","'")

    def position(self):
        return int(self.i)

    def format(self,encoding=DEFAULT_ENCODING):
        return (self.word_form()).encode(encoding)
      
    def headed(self):
        return self._head,self._drel

    def governor(self):
        """ return governing token """
        return self._gov[0]

    def governing_relation(self):
        """ return governing relation"""
        return self._gov[1]

    def dependents(self):
        """ return list of dependent tokens"""
        deps = self._deps[:]
        if self._deps == []:
            return []
        else:
            for d in self._deps:
                deps += d.dependents()
            return deps
        

    def dep_yield(self):
        """ return yield of sub-tree governed by token """
        deps = self.dependents()
        if deps == []:
            return 0
        else:
            deps.sort()
            return deps[-1].position()-deps[0].position()

    def path2root(self):
        """ returns dependency path from token to ROOT node"""
        node = self
        path = []
        while node.governor() != "ROOT":
            gov, rel = node.governor(), node.governing_relation()
            path.append( (rel, gov.mp_cat()) )
            # path.append( rel )
            node = gov
        return path

    def pws(self):
        """does the morpho tag includes a value for preceding whitespace
        if not, assume true"""
        return self.morpho.get("pws","1")=="1"

    # grammatical types
    def is_noun(self):
        return self.simple_mp_cat() == "N"

    def is_pn(self):
        return self.mp_cat() == "NPP"

    def is_pro(self):
        return self.simple_mp_cat() in ["PRO","CL"]

    def is_rel_pro(self): # qui, que, ...
        return self.mp_cat() == "PROREL" 

    def is_cl_pro(self):
        return self.simple_mp_cat() == "CL"

    def is_non_refl_cl_pro(self): # CLS
        return self.c in ["CLS","CLO"]

    def is_il(self): # ambiguous: referentiel vs. pleonastic
        return self.word_form() in ["Il","il"]

    def is_poss_det(self): # son, sa,...
        return self.mp_cat() == "DET" and self.lemma() == "son"
    
    def is_def_det(self): # l', le, la, les, au(x), ...
        return self.mp_cat() == "DET" and self.morpho.get("s",None) == "def"

    def is_indef_det(self): # un(e), quelque, aucun, certain(e)(s), ...
        return self.mp_cat() == "DET" and self.morpho.get("s",None) == "ind"

    def is_indef_pro(self): # chacun(s), tout, certain(e)(s), ...
        return self.mp_cat() == "DET" and self.morpho.get("s",None) == "ind"

    def is_dem_det(self): # ce(s), cet, ...
        return self.mp_cat() == "DET" and self.lemma() == "ce"

    def is_dem_pro(self): # celui, celui-ci, ...
        return self.mp_cat() == "PRO" and self.lemma() == "celui"

    # gender
    def gender(self):
        g = self.morpho.get("g","unk")
        if g == "unk" and self.is_pro():
            if FEM_PERS_PRO.match(self.word_form()):
                g = "f"
            elif MASC_PERS_PRO.match(self.word_form()):
                g = "m"
        return g

    # number
    def number(self):
        n = self.morpho.get("n","unk")
        if n == "unk" and self.is_pro():
            if THIRD_SG_PERS_PRO.match(self.word_form()):
                n = "s"
            elif THIRD_PL_PERS_PRO.match(self.word_form()):
                n = "p"
        return n

    def export(self,offset=None,format="LT-TTT"):
        """char offset should be given but if not, will be (word number);
        can be overridden later with lxaddis provided whitespace is handled correctly:
        next_pws is indication that next token is separated by whitespace
        """
        # pws is wrong unless tokenising trace is available
        if offset is None:
            offset=self.i
        if self.pws():
            pws="yes"
        else:
            pws="no"
        elt=ET.Element("w",{"pws":pws,"c":"w",
                            "p":self.mp_cat(),
                            "l":self.lemma(),
                            "id":"w%s"%offset})
        elt.text=self.word_form()
        return elt

    def set_sentenceId(self,sid):
        self._sentence = sid

    def sentence(self):
        return self._sentence

class DepParse( object ):
	
    def __init__(self, xml_elt):
        self._id = xml_elt.attrib["id"]
        self._tokens = {} # id => token
        self._ext2token = {} # s,e => token
        for t in xml_elt.findall("token"):
            token = Token( t )
            token.set_sentenceId(self._id)
            self._tokens[token.id] = token
            self._ext2token[token.extent()] = token
        self._deps = {}
        for tid, tok in self._tokens.items():
            gov_id, drel = tok.headed()
            assert tid != gov_id, "Loop in parse %s: %s" %(self._id,self._tokens)
            # print tid, gov_id, drel
            # self._deps[tid,gov_id] = drel
            gov = self._tokens.get(gov_id,"ROOT")
            assert (gov != "ROOT" or drel == "root"), "Drel '%s' going to ROOT!" %drel
            self._deps[tok,gov] = drel
            tok._gov = gov, drel
            if gov != "ROOT":
                gov._deps.append( tok ) 
        return

    def id(self):
        return self._id

    def tokens(self):
        return self._tokens

    def ext2token_map(self):
        return self._ext2token
    
    def get_token(self, tid):
        return self._tokens[tid]

    def get_token_from_ext(self, ext):
        return self._ext2token[ext]

    def dependencies(self):
        return self._deps




class Mention( object ):

    def __init__(self, head_token):
        self._head = head_token
        self._tokens = sorted( head_token.dependents() + [head_token] )
        self._first_token = self._tokens[0]
        self._last_token = self._tokens[-1]
        return


    def extent(self):
        s = self._first_token.extent()[0]
        e = self._last_token.extent()[-1]
        return s,e

    def text(self):
        return " ".join([t.format() for t in self._tokens])

    # ...
    def anaphoric(self):
        raise NotImplementedError
        return

    # pro methods
    def is_pro(self): 
        return self._head.is_pro()

    def is_il(self):
        return self._head.is_il()

    def is_cl_pro(self):
        return self._head.is_cl_pro()

    def is_non_refl_cl_pro(self):
        return self._head.is_non_refl_cl_pro()

    # name methods
    def is_name(self):
        return self._head.is_pn()

    def is_short_name(self, n=1):
        return self.is_name() and len(self._tokens) <= n 

    # nominal methods
    def is_def_np(self): # Q: la Marne both name and def NP
        return self._first_token.is_def_det()

    def is_short_def_np(self, n=2):
        return self.is_def_np() and len(self._tokens) <= n

    def is_dem_np(self):
        return self._first_token.is_dem_det()

    def is_short_dem_np(self, n=2):
        return self.is_dem_np() and len(self._tokens) <= n

    def is_indef_np(self):
        return self._first_token.is_indef_det()

    def is_poss_np(self):
        return self._first_token.is_poss_det()

    # lexicographic methods
    def string_match(self, other):
        return self.text() == other.text()

    def head_match(self, other):
        return self._head.text() == other._head.text()

    def name_match(self, other):
        both_names = self.is_name() and other.is_name()
        return both_names and self.string_match(other)

    def pro_match(self, other):
        both_pros = self.is_pro() and other.is_pro()
        return both_pros and self.string_match(other)

    

    # morph agreement
    def gender_agree(self, other):
        g1 = self._head.gender()
        g2 = other._head.gender()
        if g1 and g2 != None:
            return g1 == g2
        else:
            return "unk"

    def number_agree(self, other):
        g1 = self._head.number()
        g2 = other._head.number()
        if g1 != None and g2 != None:
            return g1 == g2
        else:
            return "unk"

    def morph_agree(self, other):
        g_agr = self.gender_agree(other)
        n_agr = self.number_agree(other)
        if g_agr == "unk" or n_agr == "unk":
            return "unk"
        else:
            return g_agr and n_agr

    def compat_pro_ante(self, ante):
        return self.is_pro() and self.morph_agree(ante) == True

    # misc
    def __eq__(self, other):
        return self._tokens == other._tokens

    def __lt__(self, other):
        return self.extent() < other.extent()

    def format(self,encoding=DEFAULT_ENCODING):
        return (self.word_form()).encode(encoding)

    def __repr__(self):
        return "Mention: '%s' (EXT: %s HD: %s)" %(self.text(),"%s-%s" %self.extent(),self._head)
    



class Preprocess:
    """ reads mst parse results from xml files
    eg:
    <mst_parse id="0">
    <token>
    <conll CPOSTAG="N" DEPREL="mod" FEATS="g=f|n=s" FORM="Omertà" HEAD="9" ID="1" LEMMA="omertà" PDEPREL="-" PHEAD="-" POST
    AG="NC"/>
    <extent END="6" START="0"/>
    </token>

    """

    def __init__(self,xmlfile):
        self._doc=ET.parse(xmlfile)
        self._path=xmlfile
        self._tokens = {} # extent to token map
        self._parses = defaultdict(list) # id to parse map
        for p in self._doc.findall(".//mst_parse"):
            parse = DepParse( p )
            self._parses[parse.id()] = parse
            self._tokens.update( parse.ext2token_map() )
        return

    def tokens(self):
        return self._tokens

    def parses(self):
        return self._parses

    def get_parse(self,sid):
        return self._parses[sid]
        
    def mentions(self):
        """ detect noun phrases and pronouns """
        mentions = {}
        tokens = self._tokens.values()
        for token in tokens:
            # Noun Phrases
            if token.is_noun():
                # skip Ns that are not max'l projections: e.g., "John" in "John Smith"
                # print token, token.path2root(), token.dependents(), token.dep_yield()
                gov = token.governor()
                if gov != "ROOT" and gov.is_noun():
                    continue
                m = Mention( token )
                mentions[m.extent()] = m
            # Pronouns
            elif token.is_pro():
                # skip rel. pro.
                if token.is_rel_pro():
                    continue
                # print token, token.path2root(), token.dependents(), token.dep_yield()
                m = Mention( token )
                mentions[m.extent()] = m
        return mentions


# glozz annotation reading 
# WIP !
def parse_glozz_unit(element): 
    id = element.attrib["id"]
    meta = element.find("metadata")
    e_type = element.find("characterisation/type").text
    featureset = dict([(x.attrib["name"],x.text) for x in element.findall("characterisation/featureSet/feature")])
    position = element.find("positioning")
    start = position.find("start/singlePosition").attrib["index"]
    end = position.find("end/singlePosition").attrib["index"]
    result = {"type":e_type,"attrib":{"id":id,"start":start,"end":end}}
    result["attrib"].update(featureset)
    return result

# this is a "mess in progress", should be unified with Edu model (cf also lexsim)
class EDU:
    def __init__(self,glozz_unit):
        self.attrib = glozz_unit["attrib"]
        
    def __lt__(self,other):
        s1,e1 = int(self.attrib["start"]),int(self.attrib["end"])
        s2,e2 = int(other.attrib["start"]),int(other.attrib["end"])

        # if s2<s1 and e1<e2:#1 is embedded in 2
        #     return False
        # elif s1<s2 and e2<e1:#2 is embedded in 1
        #     return True
        # else: 
        return s1<s2
            


def parse_glozz_relation(element):
    id = element.attrib["id"]
    meta = element.find("metadata")
    e_type = element.find("characterisation/type").text
    arg1 , arg2 =  element.findall("positioning/term")
    arg1 = arg1.attrib["id"]
    arg2 = arg2.attrib["id"]
    return ((arg1,arg2),e_type)


def parse_glozz_schema(element):
    id = element.attrib["id"]
    e_type = element.find("characterisation/type").text
    edus = [x.attrib["id"] for x in element.findall("positioning/embedded-unit")]
    return id,edus

#>>> xmlfile= '~/Devel/Annodis/data/annot_experts/glozz_connecteurs_aussi/fait.seg.aa'
#>>> t = annodisAnnot(xmlfile,format="glozz")
#>>> t.import_markers()
#>>> t.get_marker_text('gold_115')


#-------end of mess in progress-----------


class annodisAnnot:
    """main class for annodis annotations: discourse units + relations
    discourse units are either elementary (simple spans) or complex (set of EDUs)

    TODO: 
       - read glozz as input
    """
    def __init__(self,xmlfile,format="xml",encoding=DEFAULT_ENCODING):
        """init annotation by reading a file in one of the following
        format
        - annodis xml (one file)
        - glozz xml (2 files: aa (text) + ac (annot)
        - plain text (2 files: .seg + .rel

        todo: exports -> glozz, plain.
                 reimplement marker tagging
        """
        self._doc=ET.parse(xmlfile)
        self._path=xmlfile
        self._format = format

        if format == "xml": 
            self._edus=dict([(x.attrib["id"],x) for x in self._doc.findall(".//EDU")])
            self._cs=dict([(x.attrib["id"],x) for x in self._doc.findall(".//CS")])
            self._text=self._doc.find("text").text
            self._rels = dict([((x.find("from").text,x.find("to").text),x.find("type").text) 
                               for x in self._doc.findall(".//rel")])
        elif format=="glozz":
            base = xmlfile.split(".aa")[0]
            plain = base+".ac"
            # warning: using universal newline mode (forced, since codecs.open wont apply it with an encoding)
            self._text = self.sanitize(codecs.open(plain,encoding=encoding).read())
            self._edus = dict([(x.attrib["id"],EDU(parse_glozz_unit(x))) for x in self._doc.findall(".//unit") 
                          if x.find("characterisation/type").text=="UDE"])
            self._rels = dict([parse_glozz_relation(x) for x in self._doc.findall(".//relation")])
            self._cs = dict([parse_glozz_schema(x) for x in self._doc.findall(".//schema")])
        else:# plain text mode
            pass
        
        
    def sanitize(self,text):
        """forces universal newlines opening on codecs, by replacing \r\n and \r with \n
        """
        return text.replace("\r\n","\n").replace("\r","\n") 


    def import_markers(self,otherfile=None):
        # todo: reimplement directly into this API
        if self._format == "glozz":
            doc = self._doc
        else:
            doc=ET.parse(otherfile)

        self._markers = dict([(x.attrib["id"],(parse_glozz_unit(x))) for x in doc.findall(".//unit") 
                          if x.find("characterisation/type").text=="Connecteur"])
        print >> sys.stderr, "%d markers found"%len(self._markers)
    # todo: markers should be also injected as annotation of edus


    def markers(self):
        return self.__dict__.get("_markers",{})

    def get_marker_text(self,id):
        mark = self._markers[id]
        start,end=int(mark["attrib"]["start"]),int(mark["attrib"]["end"])
        return self.text()[start:end]

    def get_marker_span(self,id):
        mark = self._markers[id]
        start,end=int(mark["attrib"]["start"]),int(mark["attrib"]["end"])
        return start,end


    def filepath(self):
	return self._path
    
    def edus(self):
        return self._edus.values()

    def text(self):
        return self._text

    def get_edu(self,eid):
        return self._edus.get(eid)
   

    def get_rel(self,id1,id2):
        return self._rels.get((id1,id2))
    

    def get_all_rels(self):
        return self._rels
    

    def get_cdu(self,eid):
        return self._cs.get(eid)

    def get_subelements(self,cdu_id):
        """returns the ids of the EDUs making up a CDU
        """
        cdu = self.get_cdu(cdu_id)
        all = [self.get_edu(x.text) for x in cdu.findall("constituent")]
        return all


    def get_edu_text(self,eid):
        edu=self.get_edu(eid)
        if edu is not None:
            start,end=int(edu.attrib["start"]),int(edu.attrib["end"])
            return self.text()[start:end]


   
    def get_tokens(self,start,end):
        tokdict = self._prep._tokens
        result=[]
        for b,e in tokdict:
            if b>=start and e<=end:
                result.append(tokdict[b,e])
        return sorted(result)

    def get_mentions(self,start,end): 
        mdict = self._prep.mentions()
        result=[]
        for b,e in mdict:
            if b>=start and e<=end:
                result.append(mdict[b,e])
        return sorted(result)
 
    # ok TODO: exclude embedded EDU
    def get_edu_tokens(self,eid):
        edu=self.get_edu(eid)
        result = []
        if edu is not None:
            start,end=int(edu.attrib["start"]),int(edu.attrib["end"])
            result = self.get_tokens(start,end)
            exclude = []
            for one in self._edus.values():
                d,f = int(one.attrib["start"]),int(one.attrib["end"])
                if d>start and f<end:
                    exclude.extend(self.get_tokens(d,f))
            if exclude != []:
                result = list(set(result)-set(exclude))
        return result

    def get_edu_markers(self,eid):
        """ should exclude markers from embedded units """
        edu=self.get_edu(eid)
        start,end=int(edu.attrib["start"]),int(edu.attrib["end"])
        result = []
        for mark in self.markers():
            ms, me = self.get_marker_span(mark)
            if start<=ms and me<=end:
                result.append(mark)
        return result

    def get_edu_mentions(self,eid): # FIXME: some NPs stretch over several EDUs and are missed (use partial projections?)
        edu=self.get_edu(eid)
        if edu is not None:
            start,end=int(edu.attrib["start"]),int(edu.attrib["end"])
            return self.get_mentions(start,end)
        return []

    def get_parse(self,sid):
        return self._prep.parse(sid)

    def get_parse_tokens(self,sid):
        parse = self.get_parse(sid)
        if parse is not None:
            return parse # list of tokens for now
        return []

    def save(self,name):
        self._doc.write(name, encoding="utf-8")

    def name(self):
        return os.path.basename(self._path).split(".")[0]

    def add_preprocess(self,prep):
        self._prep=prep
        self._vocab=set([x.lemma() for x in prep._tokens.values()])
        self._parses = prep.parses()
        self._mentions = prep.mentions()

    def parses(self):
        return self._parses 

    def mentions(self):
        return self._mentions



     
 
class annodisCorpus:
    """ holds a collection of annodis annotations
    built from a set of files in annodis xml format
    
    todo: 
	    - include preprocessed annotations 
    """
    def __init__(self,path_to_dir,format="xml"):
        self._docs = {}
        if format == "xml":
            for a_doc in glob.glob(path_to_dir+"/*.xml"):
                if os.path.basename(a_doc).split(".",1)[1]=="xml":
	            struct = annodisAnnot(a_doc)
        	    self._docs[a_doc] = struct    
        elif format == "glozz":
            for a_doc in glob.glob(path_to_dir+"/*.aa"):
                struct = annodisAnnot(a_doc,format="glozz")
                self._docs[a_doc] = struct  
        else: 
            print >> sys.stderr, "unknown format for annodis file", format
            

    def docs(self):
	    return self._docs


    def index_on(self,field="relations"):
        """
        built different reference tables
        - field = "relations": index by relation type
        - other criteria (TODO)
	         argument pair: file,arg -> pair
        """
        self._relations = defaultdict(list)
	self._argpairs = {}
        for file,a_doc in self._docs.items():
            all_rels = a_doc.get_all_rels()
            for nodes,rel in all_rels.items():
                self._relations[rel].append((file,nodes))
		self._argpairs[(file,nodes)] = rel 

    def relations(self,relname,EDU=False):
	"""get all relations instances for relname type
	returns a set of filename+discourse unit pair
	
	if EDU is true, restrict to relations between edu pairs
	"""
	if EDU:
		return [(file,(x1,x2)) for (file,(x1,x2)) in self._relations.get(relname)
			if (self._docs[file].get_edu(x1) is not None) 
			and (self._docs[file].get_edu(x2) is not None)]
	else:
        	return self._relations.get(relname)

    def get_rel(self,file,arg1,arg2,symetric=False):
	"""find a relation between arg1 and arg2 (if any) in file
	if symetric, find a relation (1,2) or (2,1)
	"""
	direct = self._argpairs.get((file,(arg1,arg2)))
	if symetric:
		return direct or self._argpairs.get((file,(arg2,arg1)))
	else:
		return direct 

    def add_preprocess(self,suffix=".txt.prep.xml",dir="./",infile_suffix=".xml"):
	"""add all preprocessed info if available

        TODO: should work also with glozz path (.aa)
	"""
	for doc in self._docs.values():
		try:
                    basename = os.path.basename(doc.filepath()).split(infile_suffix)[0]
                    prep=Preprocess(dir+basename+suffix)
                    doc.add_preprocess(prep) #,suffix=suffix,dir=dir) 
		except:
                    print >> sys.stderr, "pb: no preprocessing for document %s when looking for %s"%(doc.filepath(), dir+basename+suffix)


    def add_markers(self,path_to_files=None,suffix=".aa"):
        """add markers from glozz annotation
        todo: check for file existence instead of simple try/except
        """
        for doc in self._docs.values():
            try:
                if path_to_files is None:
                    doc.import_markers() 
                else:
                    basename = ".".join(os.path.basename(doc.filepath()).split(".")[:-1])
                    glozz = path_to_files + "/" + basename + suffix
                    #print >> sys.stderr,  glozz
                    doc.import_markers(otherfile=glozz) 
                print >> sys.stderr, "markers found for document", doc.filepath()
            except:
                print >> sys.stderr, "--warning: no markers found for document", doc.filepath()



    def add_lexical_relations(self,table,restriction=["N","V"],table_type="voisins",stop_words=set([])):
	"""precompute lexical relations for each token 
	in corpus, according to provided table
	
	table should be indexed on combo of lemma/pos/relation type
	and yield a real value
	
	TODO: 
	x- each word should be there with itself at similarity = 1.0
	-if no restriction, all related tokens are extracted. 
	should be adjusted wrt type and location: 
		- tokens in same text
		- tokens in corpus/subcorpus
		- type: part of speech, grammatical relations if available
	"""
	self._voisins = defaultdict(list)
	self._vsn_domain = restriction
	for doc in self._docs.values():	
		vocab = set([x.lemma() for x in doc._prep._tokens.values() if x.simple_mp_cat() in restriction and not(self._voisins.has_key(x.lemma()))])
		vocab = vocab - stop_words
                print >> sys.stderr, "looking up additional %d words"%len(vocab)
		if table_type=="voisins":
                    self._voisins.update(get_voisins_dict(table,vocab))
                elif table_type=="synos":
                    self._voisins.update(get_syno_norm_dict(vocab,table))
                else:
                    print >> sys.stderr, "unimplemented lexical resource type, use 'voisins' or 'synos'", table_type
                    sys.exit(0)
                for word in vocab:
                    self._voisins[word].append((word,1.0))
        # also store rank of neighbours in each other list
        self._ranked_vsn = {}
	for entry in self._voisins:
            self._ranked_vsn[entry] = [(x[1],1.0/(i+1)) for (i,x) in enumerate(sorted([(-s,w) for (w,s) in self._voisins[entry]]))]
            self._ranked_vsn[entry] = dict(self._ranked_vsn[entry])
            self._voisins[entry]=dict(self._voisins[entry])
            #self._voisins[entry][entry]=1.0


    def word_similarity(self,word1,word2,mode="value"):
        """
        returns lexical similarity between word1 and word2 (non-symetric), either
        as a score if mode=value, or 1/rank of w2 in w1 neighbours if mode=rank
        if one is absent value is 0 and rank is 0. too (1/x)
        """
        if mode=="value":
            return self._voisins.get(word1,{}).get(word2,0.)
        else: 
            return self._ranked_vsn.get(word1,{}).get(word2,0.)

    def gen_pairs(self,sample=100,level="corpus",attach="both"):
	"""make a list of edu pairs
	- level: edus taken from diff texts from corpus ("corpus") 
	or same text ("document")
	- sample: draw sample of given size. if None, take all
	- attach: if from same doc, take related edus ("yes") or not ("no")
	or "both" (in that case, respect the distribution of pairs, so very imbalanced. for balanced samples of attach/no attach, take two samples
	"""
	result = []
        if sample is None:
            return self.gen_all_pairs()
	size = 0 
	while size < sample:
		name1,d1 = random.choice(self.docs().items())
		if level=="document":
			d2 = d1
			name2 = name1
		else:
			attach = "diff_doc"
			name2 = name1
			while name2 == name1:
				name2,d2 = random.choice(self.docs().items())
		arg1 = random.choice(d1.edus())
		arg2 = random.choice(d2.edus())
		e1id = arg1.attrib["id"]
		e2id = arg2.attrib["id"]
		if arg1!=arg2:
			relation = self.get_rel(name1,e1id,e2id,symetric=True)
			if attach=="diff_doc":
				size += 1	
				result.append((name1,e1id,name2,e2id,"diff_doc"))
			elif attach=="both":
				size += 1	
				result.append((name1,e1id,name2,e2id,str(relation)))
			elif attach=="yes":
				if relation:
					size += 1	
					result.append((name1,e1id,name2,e2id,relation))
			else:
				if relation is None:
					size += 1	
					result.append((name1,e1id,name2,e2id,"None"))
	return result


    def gen_all_pairs(self,attach="yes"):
        result = [] 
        for name,doc in self.docs().items():
            for (arg1,arg2),relation in doc.get_all_rels().items():
                if doc.get_edu(arg1) and doc.get_edu(arg2):
                    result.append((name,arg1,name,arg2,relation))
        return result


    def edu_similarity(self,d1,d2,edu1,edu2):
	pass
    




if __name__=="__main__":
    test=sys.argv[1]
    
    try:
        doc=annodisAnnot(test)
        #print doc.text().encode("utf8")
        #print doc.get_edu("2").attrib
        #print doc.get_edu_text("1")
    except:
        print >> sys.stderr, "pb reading", test
    try:
        prep=Preprocess(test.split(".xml")[0]+".txt.prep.xml")
        doc.add_preprocess(prep) 
        voisins = sys.argv[2]
	db_filename,table=init_voisins(voisins)
	corpus.add_lexical_relations(table)
        
	# for pid,p in doc.parses().items():
        #     print p.dependencies()
        # for i in range(10):
        #     print "Mentions for EDU %s: %s" %(i,doc.get_edu_mentions( str(i) ))
        mentions = doc.mentions().values()
        #mentions.sort()
        # PM: commented out for speed
        #for m in mentions:
        #    print m
        for i in range(1,10):
            try:
                print i, doc.get_edu_text( str(i) ).encode("utf8")
                print i, doc.get_edu_mentions( str(i) )
            except AttributeError:
                pass
    except:
        print >> sys.stderr, "pb reading", test.split(".xml")[0]+".txt.prep.xml"


