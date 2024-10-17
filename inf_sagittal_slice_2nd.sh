#!/bin/bash

preprocess_cmd="python preprocess_for_sagittal_stage2.py"
echo "Executing: $preprocess_cmd"
eval $preprocess_cmd

configs=("rsna_sagittal_cl")
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