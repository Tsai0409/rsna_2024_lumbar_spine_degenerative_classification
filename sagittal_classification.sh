#!/bin/bash

cmd="python preprocess_for_sagittal_classification.py"
echo "Executing: $cmd"
eval $cmd

configs=("" "")
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
