#!/bin/bash

# 2553/2778
export correction=0.9190064794816415
# attach = 1 for full expe but only unlabelled eval (to be checked)
export attach=0
export relation=1

# duplicate:
#export dupl="_dupl"
export dupl=""

export DATA_DIR=/home/phil/Devel/Annodis/data/expes_decoding
export DATA_attach_w5=$DATA_DIR/attachment_window_5_train_test${dupl}.csv
export DATA_attach_full=$DATA_DIR/attachment_full_train_test${dupl}.csv 

export DATA_4labels_w5=$DATA_DIR/relations_window5_train_test_sdrt4${dupl}.csv
export DATA_4labels_full=$DATA_DIR/relations_full_train_test_sdrt4${dupl}.csv

export DATA_18labels_w5=$DATA_DIR/relations_window5_train_test_sdrt18${dupl}.csv
export DATA_18labels_full=$DATA_DIR/relations_full_train_test_sdrt18${dupl}.csv

#HARNESS="python decoding.py"
HARNESS="attelo evaluate -C annodis.config"

LEARNERS="bayes"
# maxent"
#DECODERS="mst"
#DECODERS="local last mst astar"
#DECODERS="locallyGreedy"
#DECODERS="last local"
DECODERS="local astar"
DECODERS="local beam locallyGreedy mst"
DECODERS="nbest"

BEAM_SIZE=1
NBEST=20


learner_name() {
    if [ "$1" == "bayes" ]; then
        echo "naive bayes"
    else
        echo $1
    fi
}

run_harness() {
    msg_prefix=$1
    msg_suffix=$2
    flags=$3
    for learner in $LEARNERS; do
        echo "===== ${msg_prefix} $(learner_name $learner) $msg_suffix"
        for decoder in $DECODERS; do
	    case $decoder  in 
		"beam") decoder="astar"
		    echo "test beam" $BEAM_SIZE  
		    $HARNESS $flags -B "$BEAM_SIZE" -l "$learner" -d "$decoder" 2> logerr;
		    ;;
		"nbest") decoder="astar"
		    echo "test nbest" 1 5 10 20 100
		    $HARNESS $flags  -l "$learner" -d "$decoder" 2> logerr;
		    $HARNESS $flags -N  5 -l "$learner" -d "$decoder" 2> logerr;
		    $HARNESS $flags -N  10 -l "$learner" -d "$decoder" 2> logerr;
		    $HARNESS $flags -N  20 -l "$learner" -d "$decoder" 2> logerr;		   
		    $HARNESS $flags -N  100 -l "$learner" -d "$decoder" 2> logerr;
		    ;;
		*)  echo "$HARNESS" $flags -l "$learner" -d "$decoder" 
		    $HARNESS $flags -l "$learner" -d "$decoder" 2> logerr;
		    ;;
	    esac
        done
    done
}

if [ $attach == 1 ] ; then
    run_harness "attach"\
        "window"\
        "$DATA_attach_w5 -c $correction"
    run_harness ""\
        "windowed unlabelled eval (4)"\
        "$DATA_attach_w5 $DATA_4labels_w5 -c $correction -u"
    run_harness "attach"\
        "full _full_train_test"
        "$DATA_attach_full"
    run_harness "joint"\
        "full unlabelled evaluation (4)"\
        "$DATA_attach_full $DATA_4labels_full -u"
fi;

go() {
    pipelined=$1
    windowed=$2
    dataset=$3

    flags=""
    case "$pipelined" in
        1)  msg_prefix="pipelined"
            flags="$flags -p"
            ;;
        *)  msg_prefix="joint"
            ;;
    esac

    case "$windowed" in
        1)  DATA_ATTACH="$DATA_attach_w5"
            msg_suffix="windowed ($dataset)"
            flags="$flags -c $correction"
            ;;
        *)  DATA_ATTACH="$DATA_attach_full"
            msg_suffix="full ($dataset)"
            ;;
    esac

    case "$dataset:$windowed" in
        "4:1") DATA_REL="$DATA_4labels_w5"
            ;;
        "4:0") DATA_REL="$DATA_4labels_full"
            ;;
        "18:1") DATA_REL="$DATA_18labels_w5"
            ;;
        "18:0") DATA_REL="$DATA_18labels_full"
            ;;
        *) "Don't know how to handle dataset $dataset with windowed $windowed"
    esac

    run_harness "$msg_prefix" "$msg_suffix" "$DATA_ATTACH $DATA_REL $flags"
}

for windowed in 0; do
    for pipelined in 1; do
        for dataset in 4; do
            go  $pipelined $windowed $dataset
        done
    done
done
