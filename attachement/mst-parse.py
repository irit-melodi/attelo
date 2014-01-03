#!/usr/bin/env python

"""
mst-parse.py <dir containing raw .txt files>
"""

import sys
import os
import subprocess

# files to parse
file_ext = ".txt"
dir_path = sys.argv[1]
files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(file_ext)]
if len(files) == 0:
    sys.exit("Error: No file with extension .txt in dir %s" %dir_path)

# parse cmd
cmd = os.getenv("BONSAI")
if not cmd:
    sys.exit("Error: BONSAI var not found")
cmd += "/bin/bonsai_mst_parse_rawtext.sh"

# parse files
print >> sys.stderr, "%s file(s) to parse..." %(len(files)) 
for i,f in enumerate(files):
    print >> sys.stderr, ">>> File #", i, f
    try:
        ret_code = subprocess.call([cmd, f])
    except OSError:
        print >> sys.stderr, ">>> Warning: Problem in parsing file %s" %f

