#!/bin/bash

cmd="python preprocess_for_sagittal_stage1.py"
echo "Executing: $cmd"
eval $cmd

configs=("rsna_sagittal_level_cl_spinal_v1" "rsna_sagittal_level_cl_nfn_v1")
folds=(0 1 2 3 4)

for config in "${configs[@]}"
do
    for fold in "${folds[@]}"
    do
        cmd="python train_one_fold.py -c $config -f $fold"
        echo "Executing: $cmd"
        eval $cmd
        infcmd="python predict.py -c $config -f $fold"
        echo "Executing: $infcmd"
        eval $infcmd        
        echo "----------------------------------------"
    done
done