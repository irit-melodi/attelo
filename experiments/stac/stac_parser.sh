#!/bin/bash

set -e

#test cross-validation avec attachement et relations
#
DATA_DIR=/tmp/charette-2013-12-30
DATA_EXT=.csv # .2.csv
DECODE_FLAGS="-C stac"

python  decoding.py $DECODE_FLAGS $DATA_DIR/pilot.edu-pairs$DATA_EXT        $DATA_DIR/pilot.relations$DATA_EXT        -l bayes -d mst
python  decoding.py $DECODE_FLAGS $DATA_DIR/socl-season1.edu-pairs$DATA_EXT $DATA_DIR/socl-season1.relations$DATA_EXT -l bayes -d mst

# test stand-alone parser for stac
# 1) train and save attachment model
# -i
python decoding.py $DECODE_FLAGS -S $DATA_DIR/pilot.edu-pairs$DATA_EXT -l bayes
# 2) predict attachment (same instances here, but should be sth else) 
# NB: updated astar decoder seems to fail / TODO: check with the real subdoc id
# -i
python decoding.py $DECODE_FLAGS -T -A attach.model -o tmp $DATA_DIR/pilot.edu-pairs$DATA_EXT -d mst

# attahct + relations: TODO: relation file is not generated properly yet
# 1b) train + save attachemtn+relations models
python  decoding.py $DECODE_FLAGS -S $DATA_DIR/pilot.edu-pairs$DATA_EXT $DATA_DIR/pilot.relations$DATA_EXT -l bayes

# 2b) predict attachment + relations
# -i
python decoding.py $DECODE_FLAGS -T -A attach.model -R relations.model -o tmp/ $DATA_DIR/pilot.edu-pairs$DATA_EXT $DATA_DIR/pilot.relations$DATA_EXT -d mst



# results
#socl
#FINAL EVAL: relations full: 	 locallyGreedy+bayes, h=average, unlabelled=False,post=False,rfc=full 	 Prec=0.229, Recall=0.217, F1=0.223 +/- 0.015 (0.239 +- 0.029)
#FINAL EVAL: relations full: 	 local+maxent, h=average, unlabelled=False,post=False,rfc=full 	         Prec=0.678, Recall=0.151, F1=0.247 +/- 0.017 (0.243 +- 0.034)
#FINAL EVAL: relations full: 	 local+bayes, h=average, unlabelled=False,post=False,rfc=full 	                 Prec=0.261, Recall=0.249, F1=0.255 +/- 0.015 (0.264 +- 0.031)
#FINAL EVAL: relations full: 	 locallyGreedy+maxent, h=average, unlabelled=False,post=False,rfc=full 	 Prec=0.281, Recall=0.257, F1=0.269 +/- 0.015 (0.277 +- 0.030)

#pilot
#FINAL EVAL: relations full  : 	 locallyGreedy+maxent, h=average, unlabelled=False,post=False,rfc=full 	 Prec=0.341, Recall=0.244, F1=0.284 +/- 0.015 (0.279 +- 0.029)
