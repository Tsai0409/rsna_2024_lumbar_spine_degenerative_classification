# RSNA 2024 Lumbar Spine Degenerative Classification 2nd Place Solution
This repository contains the training code for Yuji's part. To reproduce the 2nd place solution, it's also necessary to reproduce Ian's and Bartley's solutions. The final prediction value is calculated using a simple weighted average:
```
(Yuji's prediction * 2 + Ian's prediction * 2 + Bartley's prediction) / 5
```

# 1, data download
```
download_competition_data.sh
```

# 2, axial level estimation
Estimate the slices to be used for predicting 5 levels. I'm referring to hengck23's code. Execute the following code in the Kaggle environment, not in your own environment, and download the output.
https://www.kaggle.com/code/yujiariyasu/axial-level-estimation
```
kaggle kernels output yujiariyasu/axial-level-estimation -p ./input/
```

# 3, dicom to png
```
python dcm_to_png.py
```

# 4, preprocess
```
python preprocess.py
```

# 5, sagittal slice estimation 
```
. ./inf_sagittal_slice_1st_stage.sh
. ./inf_sagittal_slice_2nd_stage.sh
```

# 6, region estimation by yolox
```
. ./axial_yolo.sh
. ./sagittal_yolo.sh
```

# 7, axial classification
```
. ./axial_classification.sh
```

# 8, sagittal classification
```
. ./sagittal_classification.sh
```

# 9, noise reduction by oof
In addition to my own oof, I used Ian's and Bartley's oof. In the code, I used the pre-calculated oof.
```
python find_noisy_label.py
```
Note that the label noise data is also used for Ian's training.


# 10, retrain by clean label data
```
. ./axial_classification_by_clean_data.sh
. ./sagittal_classification_by_clean_data.sh
```

# 11, inference in kaggle notebook
The trained model needs to be uploaded to kaggle.
Also, it is necessary to correctly set the model weights by referencing the config name. In the notebook, pre-calculated model weights are used.
