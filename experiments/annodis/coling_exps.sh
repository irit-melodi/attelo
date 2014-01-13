#!/bin/bash
# NB: désolé pour le bourrinage (PM)

# 2553/2778
export correction=0.9190064794816415
# 
export attach=0
export relation=2

# duplicate:
export dupl="_dupl"
#export dupl=""

export DATA_attach_w5=../../../data/expes_decoding/attachment_window_5_train_test${dupl}.csv
export DATA_attach_full=../../../data/expes_decoding/attachment_full_train_test${dupl}.csv 

export DATA_4labels_w5=../../../data/expes_decoding/relations_window5_train_test_sdrt4${dupl}.csv
export DATA_4labels_full=../../../data/expes_decoding/relations_full_train_test_sdrt4${dupl}.csv

export DATA_18labels_w5=../../../data/expes_decoding/relations_window5_train_test_sdrt18${dupl}.csv
export DATA_18labels_full=../../../data/expes_decoding/relations_full_train_test_sdrt18${dupl}.csv



if [ $attach == 1 ] ; then
 echo  "===== attach naive bayes windowed"
  python decoding.py $DATA_attach_w5 -c $correction    2> logerr ;
  python decoding.py $DATA_attach_w5  -c $correction  -d mst 2> logerr ;
  python decoding.py $DATA_attach_w5  -c $correction    -d astar 2> logerr ;
 echo "====== attach maxent window"
 python decoding.py $DATA_attach_w5 -c $correction   -l maxent 2> logerr ;
 python decoding.py $DATA_attach_w5  -c $correction  -l maxent -d last 2> logerr ;
 python decoding.py $DATA_attach_w5 -c $correction   -l maxent -d mst 2> logerr ;
 python decoding.py $DATA_attach_w5 -c $correction   -l maxent -d astar 2> logerr ;

 echo "======  naive bayes windowed unlabelled eval (4)"
 python decoding.py $DATA_attach_w5 $DATA_4labels_w5 -c $correction -u  2> logerr ;
 python decoding.py $DATA_attach_w5 $DATA_4labels_w5 -c $correction -u   -d mst 2> logerr ;
 python decoding.py $DATA_attach_w5 $DATA_4labels_w5 -c $correction  -u   -d astar 2> logerr ;
 echo "======  maxent windowed unlabelled eval (4)"
 python decoding.py $DATA_attach_w5 $DATA_4labels_w5 -c $correction  -u -l maxent 2> logerr ;
 python decoding.py $DATA_attach_w5 $DATA_4labels_w5 -c $correction -u  -l maxent -d last 2> logerr ;
 python decoding.py $DATA_attach_w5 $DATA_4labels_w5 -c $correction  -u  -l maxent -d mst 2> logerr ;
 python decoding.py $DATA_attach_w5 $DATA_4labels_w5 -c $correction   -u -l maxent -d astar 2> logerr ;


 echo "====== attach naive bayes full _full_train_test"
  python decoding.py $DATA_attach_full    2> logerr ;
  python decoding.py $DATA_attach_full   -d mst 2> logerr ;
  python decoding.py $DATA_attach_full   -d astar 2> logerr ;
 echo "====== attach maxent full"
 python decoding.py $DATA_attach_full   -l maxent 2> logerr ;
 python decoding.py $DATA_attach_full   -l maxent -d last 2> logerr ;
 python decoding.py $DATA_attach_full   -l maxent -d mst 2> logerr ;
 python decoding.py $DATA_attach_full   -l maxent -d astar 2> logerr ;

echo "====== joint naive bayes full unlabelled evaluation (4)"
python decoding.py $DATA_attach_full $DATA_4labels_full -u   2> logerr ;
python decoding.py $DATA_attach_full $DATA_4labels_full -u    -d mst 2> logerr ;
python decoding.py $DATA_attach_full $DATA_4labels_full -u     -d astar 2> logerr ;
 echo "====== joint maxent full unlabelled evaluation (4)"
python decoding.py $DATA_attach_full $DATA_4labels_full -u    -l maxent 2> logerr ;
python decoding.py $DATA_attach_full $DATA_4labels_full -u   -l maxent -d last 2> logerr ;
python decoding.py $DATA_attach_full $DATA_4labels_full  -u   -l maxent -d mst 2> logerr ;
python decoding.py $DATA_attach_full $DATA_4labels_full -u    -l maxent -d astar 2> logerr ;

fi;

# locallyGreedy !
if [ $relation == 2 ] ; then 
 python decoding.py $DATA_attach_w5 $DATA_4labels_w5 -c $correction  -d locallyGreedy 2> logerr ;
 python decoding.py $DATA_attach_w5 $DATA_4labels_w5 -c $correction  -d locallyGreedy -l maxent 2> logerr ;

 python decoding.py $DATA_attach_w5 $DATA_18labels_w5 -c $correction  -d locallyGreedy  2> logerr ;

 echo "======  maxent windowed (18)"
 python decoding.py $DATA_attach_w5 $DATA_18labels_w5 -c $correction -d locallyGreedy  -l maxent 2> logerr ;


 echo "====== pipelined naive bayes windowed (4)"
 python decoding.py $DATA_attach_w5 $DATA_4labels_w5 -c $correction -d locallyGreedy -p  2> logerr ;

 echo "====== pipelined maxent windowed (4)"
 python decoding.py $DATA_attach_w5 $DATA_4labels_w5 -c $correction -d locallyGreedy -p -l maxent 2> logerr ;



 echo "====== pipelined naive bayes windowed (18)"
 python decoding.py $DATA_attach_w5 $DATA_18labels_w5 -c $correction -d locallyGreedy -p  2> logerr ;

 echo "====== pipelined maxent windowed (18)"
 python decoding.py $DATA_attach_w5 $DATA_18labels_w5 -c $correction -d locallyGreedy -p -l maxent 2> logerr ;





 echo "====== joint naive bayes full (4)"
 python decoding.py $DATA_attach_full $DATA_4labels_full  -d locallyGreedy  2> logerr ;

 echo "====== joint maxent full (4)"
python decoding.py $DATA_attach_full $DATA_4labels_full -d locallyGreedy  -l maxent 2> logerr ;


 
 echo "====== pipelined naive bayes full (4)"
 python decoding.py $DATA_attach_full $DATA_4labels_full -d locallyGreedy -p   2> logerr ;

 echo "====== pipelined maxent full (4)"
python decoding.py $DATA_attach_full $DATA_4labels_full -d locallyGreedy -p   -l maxent 2> logerr ;



 echo "====== joint naive bayes full (18)"
 python decoding.py $DATA_attach_full $DATA_18labels_full  -d locallyGreedy  2> logerr ;

 echo "====== joint maxent full (18)"
python decoding.py $DATA_attach_full $DATA_18labels_full  -d locallyGreedy -l maxent 2> logerr ;


 echo "====== pipelined naive bayes full (18)"
 python decoding.py $DATA_attach_full $DATA_18labels_full -d locallyGreedy -p   2> logerr ;

 echo "====== pipelined maxent full (18)"
python decoding.py $DATA_attach_full $DATA_18labels_full -d locallyGreedy -p   -l maxent 2> logerr ;


fi;


if [ $relation == 1 ] ; then 

 echo "======  naive bayes windowed (4)"
 python decoding.py $DATA_attach_w5 $DATA_4labels_w5 -c $correction   2> logerr ;
 python decoding.py $DATA_attach_w5 $DATA_4labels_w5 -c $correction   -d mst 2> logerr ;
 python decoding.py $DATA_attach_w5 $DATA_4labels_w5 -c $correction    -d astar 2> logerr ;
 echo "======  maxent windowed (4)"
 python decoding.py $DATA_attach_w5 $DATA_4labels_w5 -c $correction  -l maxent 2> logerr ;
 python decoding.py $DATA_attach_w5 $DATA_4labels_w5 -c $correction  -l maxent -d last 2> logerr ;
 python decoding.py $DATA_attach_w5 $DATA_4labels_w5 -c $correction   -l maxent -d mst 2> logerr ;
 python decoding.py $DATA_attach_w5 $DATA_4labels_w5 -c $correction   -l maxent -d astar 2> logerr ;


 echo "======  naive bayes windowed (18)"
 python decoding.py $DATA_attach_w5 $DATA_18labels_w5 -c $correction   2> logerr ;
 python decoding.py $DATA_attach_w5 $DATA_18labels_w5 -c $correction   -d mst 2> logerr ;
 python decoding.py $DATA_attach_w5 $DATA_18labels_w5 -c $correction    -d astar 2> logerr ;
 echo "======  maxent windowed (18)"
 python decoding.py $DATA_attach_w5 $DATA_18labels_w5 -c $correction  -l maxent 2> logerr ;
 python decoding.py $DATA_attach_w5 $DATA_18labels_w5 -c $correction  -l maxent -d last 2> logerr ;
 python decoding.py $DATA_attach_w5 $DATA_18labels_w5 -c $correction   -l maxent -d mst 2> logerr ;
 python decoding.py $DATA_attach_w5 $DATA_18labels_w5 -c $correction   -l maxent -d astar 2> logerr ;

 echo "====== pipelined naive bayes windowed (4)"
 python decoding.py $DATA_attach_w5 $DATA_4labels_w5 -c $correction -p  2> logerr ;
 python decoding.py $DATA_attach_w5 $DATA_4labels_w5 -c $correction -p  -d mst 2> logerr ;
 python decoding.py $DATA_attach_w5 $DATA_4labels_w5 -c $correction  -p  -d astar 2> logerr ;
 echo "====== pipelined maxent windowed (4)"
 python decoding.py $DATA_attach_w5 $DATA_4labels_w5 -c $correction -p -l maxent 2> logerr ;
 python decoding.py $DATA_attach_w5 $DATA_4labels_w5 -c $correction -p -l maxent -d last 2> logerr ;
 python decoding.py $DATA_attach_w5 $DATA_4labels_w5 -c $correction  -p -l maxent -d mst 2> logerr ;
 python decoding.py $DATA_attach_w5 $DATA_4labels_w5 -c $correction  -p -l maxent -d astar 2> logerr ;


 echo "====== pipelined naive bayes windowed (18)"
 python decoding.py $DATA_attach_w5 $DATA_18labels_w5 -c $correction -p  2> logerr ;
 python decoding.py $DATA_attach_w5 $DATA_18labels_w5 -c $correction -p  -d mst 2> logerr ;
 python decoding.py $DATA_attach_w5 $DATA_18labels_w5 -c $correction  -p  -d astar 2> logerr ;
 echo "====== pipelined maxent windowed (18)"
 python decoding.py $DATA_attach_w5 $DATA_18labels_w5 -c $correction -p -l maxent 2> logerr ;
 python decoding.py $DATA_attach_w5 $DATA_18labels_w5 -c $correction -p -l maxent -d last 2> logerr ;
 python decoding.py $DATA_attach_w5 $DATA_18labels_w5 -c $correction  -p -l maxent -d mst 2> logerr ;
 python decoding.py $DATA_attach_w5 $DATA_18labels_w5 -c $correction  -p -l maxent -d astar 2> logerr ;




 echo "====== joint naive bayes full (4)"
 python decoding.py $DATA_attach_full $DATA_4labels_full    2> logerr ;
 python decoding.py $DATA_attach_full $DATA_4labels_full    -d mst 2> logerr ;
 python decoding.py $DATA_attach_full $DATA_4labels_full     -d astar 2> logerr ;
 echo "====== joint maxent full (4)"
python decoding.py $DATA_attach_full $DATA_4labels_full   -l maxent 2> logerr ;
python decoding.py $DATA_attach_full $DATA_4labels_full   -l maxent -d last 2> logerr ;
python decoding.py $DATA_attach_full $DATA_4labels_full    -l maxent -d mst 2> logerr ;
python decoding.py $DATA_attach_full $DATA_4labels_full    -l maxent -d astar 2> logerr ;

 
 echo "====== pipelined naive bayes full (4)"
 python decoding.py $DATA_attach_full $DATA_4labels_full  -p   2> logerr ;
 python decoding.py $DATA_attach_full $DATA_4labels_full  -p   -d mst 2> logerr ;
 python decoding.py $DATA_attach_full $DATA_4labels_full  -p   -d astar 2> logerr ;
 echo "====== pipelined maxent full (4)"
python decoding.py $DATA_attach_full $DATA_4labels_full -p   -l maxent 2> logerr ;
python decoding.py $DATA_attach_full $DATA_4labels_full -p   -l maxent -d last 2> logerr ;
python decoding.py $DATA_attach_full $DATA_4labels_full -p   -l maxent -d mst 2> logerr ;
python decoding.py $DATA_attach_full $DATA_4labels_full -p    -l maxent -d astar 2> logerr ;


 echo "====== joint naive bayes full (18)"
 python decoding.py $DATA_attach_full $DATA_18labels_full    2> logerr ;
 python decoding.py $DATA_attach_full $DATA_18labels_full    -d mst 2> logerr ;
 python decoding.py $DATA_attach_full $DATA_18labels_full     -d astar 2> logerr ;
 echo "====== joint maxent full (18)"
python decoding.py $DATA_attach_full $DATA_18labels_full   -l maxent 2> logerr ;
python decoding.py $DATA_attach_full $DATA_18labels_full   -l maxent -d last 2> logerr ;
python decoding.py $DATA_attach_full $DATA_18labels_full    -l maxent -d mst 2> logerr ;
python decoding.py $DATA_attach_full $DATA_18labels_full    -l maxent -d astar 2> logerr ;

 echo "====== pipelined naive bayes full (18)"
 python decoding.py $DATA_attach_full $DATA_18labels_full  -p   2> logerr ;
 python decoding.py $DATA_attach_full $DATA_18labels_full  -p   -d mst 2> logerr ;
 python decoding.py $DATA_attach_full $DATA_18labels_full  -p   -d astar 2> logerr ;
 echo "====== pipelined maxent full (18)"
python decoding.py $DATA_attach_full $DATA_18labels_full -p   -l maxent 2> logerr ;
python decoding.py $DATA_attach_full $DATA_18labels_full -p   -l maxent -d last 2> logerr ;
python decoding.py $DATA_attach_full $DATA_18labels_full -p   -l maxent -d mst 2> logerr ;
python decoding.py $DATA_attach_full $DATA_18labels_full -p    -l maxent -d astar 2> logerr ;

fi;
