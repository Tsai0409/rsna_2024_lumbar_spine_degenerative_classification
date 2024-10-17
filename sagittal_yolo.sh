#!/bin/bash

cd input
kaggle datasets download -d yujiariyasu/bartley-coords-rsna-improved-csv
unzip bartley-coords-rsna-improved-csv.zip
cd ..

cmd="python preprocess_for_sagittal_yolo.py"
echo "Executing: $cmd"
eval $cmd

configs=("rsna_10classes_yolox_x")
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