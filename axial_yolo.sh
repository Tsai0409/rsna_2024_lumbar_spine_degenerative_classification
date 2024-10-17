#!/bin/bash

cmd="python preprocess_for_axial_yolo.py"
echo "Executing: $cmd"
eval $cmd

configs=("rsna_axial_all_images_left_yolox_x" "rsna_axial_all_images_right_yolox_x")
folds=(0 1 2 3 4)

for config in "${configs[@]}"
do
    for fold in "${folds[@]}"
    do
        cmd="python yolox_train_one_fold.py -c $config -f $fold"
        echo "Executing: $cmd"
        eval $cmd
        echo "----------------------------------------"
    done
done