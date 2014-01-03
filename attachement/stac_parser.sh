#test cross-validation avec attachement et relations
#
python  decoding.py /home/phil/Devel/Stac/expes/charrette-2013-07-25T1530/pilot.edu-pairs.2.csv /home/phil/Devel/Stac/expes/charrette-2013-07-25T1530/pilot.relations.2.csv -l bayes -d mst

python  decoding.py /home/phil/Devel/Stac/expes/charrette-2013-07-25T1530/socl-season1.edu-pairs.2.csv /home/phil/Devel/Stac/expes/charrette-2013-07-25T1530/socl-season1.relations.2.csv -l bayes -d mst

# test stand-alone parser for stac
# 1) train and save attachment model
python -i  decoding.py -S /home/phil/Devel/Stac/expes/charrette-2013-06-26T1644Z/pilot.edu-pairs.2.csv -l bayes
# 2) predict attachment (same instances here, but should be sth else) 
# NB: updated astar decoder seems to fail / TODO: check with the real subdoc id
python -i  decoding.py -T -A attach.model -o tmp /home/phil/Devel/Stac/expes/charrette-2013-06-26T1644Z/pilot.edu-pairs.2.csv -d mst

# attahct + relations: TODO: relation file is not generated properly yet
# 1b) train + save attachemtn+relations models
python  decoding.py -S /home/phil/Devel/Stac/expes/charrette-2013-07-25T1530/pilot.edu-pairs.2.csv /home/phil/Devel/Stac/expes/charrette-2013-07-25T1530/pilot.relations.2.csv -l bayes

# 2b) predict attachment + relations
python -i  decoding.py -T -A attach.model -R relations.model -o tmp/ /home/phil/Devel/Stac/expes/charrette-2013-07-25T1530/pilot.edu-pairs.2.csv /home/phil/Devel/Stac/expes/charrette-2013-07-25T1530/pilot.relations.2.csv -d mst



# results
#socl
#FINAL EVAL: relations full: 	 locallyGreedy+bayes, h=average, unlabelled=False,post=False,rfc=full 	 Prec=0.229, Recall=0.217, F1=0.223 +/- 0.015 (0.239 +- 0.029)
#FINAL EVAL: relations full: 	 local+maxent, h=average, unlabelled=False,post=False,rfc=full 	         Prec=0.678, Recall=0.151, F1=0.247 +/- 0.017 (0.243 +- 0.034)
#FINAL EVAL: relations full: 	 local+bayes, h=average, unlabelled=False,post=False,rfc=full 	                 Prec=0.261, Recall=0.249, F1=0.255 +/- 0.015 (0.264 +- 0.031)
#FINAL EVAL: relations full: 	 locallyGreedy+maxent, h=average, unlabelled=False,post=False,rfc=full 	 Prec=0.281, Recall=0.257, F1=0.269 +/- 0.015 (0.277 +- 0.030)

#pilot
#FINAL EVAL: relations full  : 	 locallyGreedy+maxent, h=average, unlabelled=False,post=False,rfc=full 	 Prec=0.341, Recall=0.244, F1=0.284 +/- 0.015 (0.279 +- 0.029)
