# preprocess_20holdout.py
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import pandas as pd
from sklearn.model_selection import train_test_split

# kaggle input
DATA_KAGGLE_DIR = "/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification"

# 設定環境變數
WORKING_DIR="/kaggle/working/duplicate"

# 讀 sagittal + axial
sa = pd.read_csv(f'{WORKING_DIR}/csv_train/dcm_to_png_3/sagittal_df.csv')
ax = pd.read_csv(f'{WORKING_DIR}/csv_train/dcm_to_png_3/axial_df.csv')
del ax['z']
pdf = pd.concat([sa, ax])

# 讀標籤
label_df = pd.read_csv(f'{DATA_KAGGLE_DIR}/train.csv')
df = pd.read_csv(f'{DATA_KAGGLE_DIR}/train_series_descriptions.csv')
cood = pd.read_csv(f'{DATA_KAGGLE_DIR}/train_label_coordinates.csv')

# 先挑出標籤欄位
label_features = []
for c in list(label_df):
    if c == 'study_id':
        continue
    if len(label_df[label_df[c]=='Severe']) < 30:
        label_features.append(c)

# =========================
# 先切 20% holdout
# =========================
# 只要保證相同 study_id，不要分散
study_ids = label_df['study_id'].unique()
holdout_ids, train_ids = train_test_split(
    study_ids,
    test_size=0.8,
    random_state=42,
    shuffle=True
)

# 建立 holdout dataframe
holdout_df = label_df[label_df['study_id'].isin(holdout_ids)].copy()
holdout_df['fold'] = -1  # -1 表示永遠 holdout

# 接下來做 5 fold 只針對 80% train_ids
train_df = label_df[label_df['study_id'].isin(train_ids)].copy()

# 建立 one-hot
one_hot_labels = train_df[label_features].fillna('Normal/Mild').values

mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=2021)
for fold, (train_index, val_index) in enumerate(mskf.split(one_hot_labels, one_hot_labels)):
    train_df.loc[train_df.iloc[val_index].index, 'fold'] = fold

# 合併回來
final_label_df = pd.concat([train_df, holdout_df])

# =========================
# 輸出
# =========================
train = pdf.merge(final_label_df, on='study_id').merge(df, on=['study_id', 'series_id'])
train['fold'] = train['fold'].astype(int)

train.to_csv('train_with_fold.csv', index=False)

# 另外存一個 holdout 名單
holdout_df[['study_id']].drop_duplicates().to_csv('holdout_test_study_ids.csv', index=False)

print("✅ 完成，train_with_fold.csv 中 fold=-1 即為 holdout。")

print("\n✅ 各 fold 資料量統計：")
print(train['fold'].value_counts().sort_index())

print("\n✅ 各 fold 對應的唯一 study_id 數量：")
print(train.groupby("fold")["study_id"].nunique().sort_index())

studyid_counts = train.groupby("fold")["study_id"].nunique().sort_index()
total_studies = studyid_counts.sum()
print("\n✅ 各 fold 對應的唯一 study_id 數量 & 百分比：")
for fold, count in studyid_counts.items():
    percent = (count / total_studies) * 100
    print(f"  fold {fold:>2}: {count} studies ({percent:.2f}%)")
