#!/bin/bash
export DATA_attach_w5=../../../data/expes_decoding/attachment_window_5_train_test.csv
export DATA_attach_full=../../../data/expes_decoding/attachment_full_train_test.csv 

export DATA_4labels_w5=../../../data/expes_decoding/relations_window5_train_test_sdrt4.csv
export DATA_4labels_full=../../../data/expes_decoding/relations_full_train_test_sdrt4.csv

export DATA_18labels_w5=../../../data/expes_decoding/relations_window5_train_test_sdrt18.csv
export DATA_18labels_full=../../../data/expes_decoding/relations_full_train_test_sdrt18.csv

for f in $DATA_attach_w5 $DATA_attach_full $DATA_4labels_w5 $DATA_4labels_full $DATA_18labels_w5 $DATA_18labels_full; 
do
echo "doing" $f "->" $(basename $f .csv)_dupl.csv; 
python  duplicateFeatures.py $f 'D#SAME_SENTENCE' $(basename $f .csv)_dupl.csv; 
done;