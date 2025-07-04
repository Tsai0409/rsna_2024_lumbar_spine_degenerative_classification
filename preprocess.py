# preprocess.py
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import pandas as pd

# kaggle input
DATA_KAGGLE_DIR = "/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification"

# 設定環境變數
WORKING_DIR="/kaggle/working/duplicate"

# sa = pd.read_csv('input/sagittal_df.csv')
sa = pd.read_csv(f'{WORKING_DIR}/dcm_to_png_3/sagittal_df.csv')
# ax = pd.read_csv('input/axial_df.csv')
ax = pd.read_csv(f'{WORKING_DIR}/dcm_to_png_3/axial_df.csv')
del ax['z']
pdf = pd.concat([sa, ax])

# label_df = pd.read_csv('input/train.csv')
label_df = pd.read_csv(f'{DATA_KAGGLE_DIR}/train.csv')
# df = pd.read_csv('input/train_series_descriptions.csv')
df = pd.read_csv(f'{DATA_KAGGLE_DIR}/train_series_descriptions.csv')
# cood = pd.read_csv('input/train_label_coordinates.csv')
cood = pd.read_csv(f'{DATA_KAGGLE_DIR}/train_label_coordinates.csv')

label_features = []
for c in list(label_df):
    if c == 'study_id':
        continue
    if len(label_df[label_df[c]=='Severe']) < 30:
        label_features.append(c)  # 之後做 stratified split

one_hot_labels = label_df[label_features].fillna('Normal/Mild').values
mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=2021)  # 建立一個多標籤分層交叉驗證 (MultilabelStratifiedKFold) 的對象，將資料分成 5 折，並使用隨機打亂資料（以 random_state=2021 保證結果可重現）

for fold, (train_index, val_index) in enumerate(mskf.split(one_hot_labels, one_hot_labels)):  # 為每一折分配 fold 標籤
    label_df.loc[val_index, 'fold'] = int(fold)
label_df.fold.value_counts()

train = pdf.merge(label_df, on='study_id').merge(df, on=['study_id', 'series_id'])
train['fold'] = train['fold'].astype(int)
# train.to_csv('input/train_with_fold.csv', index=False)
train.to_csv('train_with_fold.csv', index=False)

print("label_features = ", label_features)

print("✅ stratified 會使用的 label_features 以及 Severe 樣本數：")
for c in label_features:
    severe_count = (label_df[c] == 'Severe').sum()
    print(f" - {c}: Severe 數量 = {severe_count}")

print("✅ 所有標籤 Severe 樣本數統計：")
for c in label_df.columns:
    if c == "study_id":
        continue
    severe_count = (label_df[c] == "Severe").sum()
    print(f" - {c}: Severe 數量 = {severe_count}")

print('preprocess.py finish')