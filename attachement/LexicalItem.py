class LexicalItem:
    """
    class for interfacing lexical entries in a lexicon
    """

    # correspondance for pos tags, for normalisation purposes
    _cat_norm={"S":"N",
               }

    def __init__(self,cat,lemma,form=None):
        self._cat=self._cat_norm.get(cat,cat)
        self._lemma=lemma
        self._form=form

    def cat(self):
        return self._cat

    def lemma(self):
        return self._lemma

    def form(self):
        return self._form

    def __repr__(self):
        return ("%s.%s"%(self.cat(),self.lemma())).decode("latin1")

    def __str__(self):
        return ("%s.%s"%(self.cat(),self.lemma()))

    def __eq__(self,other):
        return self.cat()==other.cat() and self.lemma()==other.lemma()

