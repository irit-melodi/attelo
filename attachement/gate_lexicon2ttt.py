#!/usr/bin/env python
# -*- coding: utf-8 -*-
  
"""converts a directory (ARG1) of Gate lexicon to LT-TTT format in 
another directory (ARG2)

ARG2 must already exist
"""

from Lookup import Gazetteer
import sys

indir=sys.argv[1]
outdir=sys.argv[2]
lexicon=Gazetteer(indir)
lexicon.export(directory=outdir,target="LT-TTT")
