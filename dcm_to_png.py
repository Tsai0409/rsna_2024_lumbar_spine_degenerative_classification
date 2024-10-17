import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from glob import glob
import albumentations as A
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import copy
from scipy.special import softmax
def sigmoid(x):
    return 1/(1 + np.exp(-x))
from sklearn.metrics import roc_auc_score, confusion_matrix, mean_squared_error, average_precision_score, recall_score
import warnings
warnings.simplefilter('ignore')

import os
import cv2
import gdcm
import pydicom
import zipfile
import dicomsdl
from tqdm import tqdm

def standardize_pixel_array(dcm: pydicom.dataset.FileDataset) -> np.ndarray:
    """
    Source : https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/427217
    """
    # Correct DICOM pixel_array if PixelRepresentation == 1.
    pixel_array = dcm.pixel_array
    if dcm.PixelRepresentation == 1:
        bit_shift = dcm.BitsAllocated - dcm.BitsStored
        dtype = pixel_array.dtype 
        new_array = (pixel_array << bit_shift).astype(dtype) >>  bit_shift
        pixel_array = pydicom.pixel_data_handlers.util.apply_modality_lut(new_array, dcm)
    return pixel_array

def get_center_x_path(study_id, series_id):
    path_x_map = {}
    for dcm_path in sorted(glob(f"{dicom_dir}/{study_id}/{series_id}/*.dcm")):
        filename = dcm_path.split('/')[-1].replace('.dcm', '')
    #         if int(instance_number) % 2 == 1:
    #             continue
        dicom = dicomsdl.open(dcm_path)
        pos_x = dicom['ImagePositionPatient'][0]
        path_x_map[pos_x] = dcm_path
        xs = []
        for i, k in enumerate(sorted(path_x_map.keys())):
            xs.append(k)
    return path_x_map[xs[len(xs)//2-1]], path_x_map[xs[len(xs)//2]], path_x_map[xs[len(xs)//2+1]]

def read_sagittal_x_center_dicom(args, verbose=False):
    study_id, series_id = args
    ch_imgs = []
    dcm_path_3 = get_center_x_path(study_id, series_id)
    for dcm_path in dcm_path_3:
        dicom = dicomsdl.open(dcm_path)
        img = dicom.pixelData(storedvalue = True)

        if dicom['PixelRepresentation'] == 1:
            bit_shift = dicom['BitsAllocated'] - dicom['BitsStored']
            dtype = img.dtype
            img = (img << bit_shift).astype(dtype) >>  bit_shift
        img = img.astype(np.float32)

        intercept = dicom['RescaleIntercept']
        slope = dicom['RescaleSlope']
        if (slope is not None) & (intercept is not None):
            img = img * slope + intercept

        img = (img - img.min()) / (img.max() - img.min() + 1e-6)

        if dicom.PhotometricInterpretation == "MONOCHROME1":
            img = 1 - img
        img = (img*255.0).astype('uint8')

        ch_imgs.append(img)
    if verbose:
        plt.imshow(ch_imgs[1], 'gray')
        plt.show()
    img = np.array(ch_imgs).transpose((1,2,0))
    cv2.imwrite(f'input/sagittal_all_images/{study_id}___{series_id}.png', img)
    
def select_path_list(lst, N, offset=0, skip=1):
    if not lst:
        return [''] * (2 * N + 1)
    
    center = (len(lst) // 2) + offset
    result = [''] * (2 * N + 1)
    
    if 0 <= center < len(lst):
        result[N] = lst[center]
    
    for i in range(1, N + 1):
        left_index = center - i*skip
        right_index = center + i*skip
        
        if 0 <= left_index < len(lst):
            result[N - i] = lst[left_index]
        else:
            result[N - i] = lst[0]
        if 0 <= right_index < len(lst):
            result[N + i] = lst[right_index]
        else:
            result[N + i] = lst[len(lst)-1]
    
    # st()
    return result

    
def read_sagittal_dicom(args, verbose=False):
    study_id, series_id = args
    imgs = {}
    origin_paths = []
    zs = []
    xyzs = []
    paths = []
    for dcm_path in glob(f"{dicom_dir}/{study_id}/{series_id}/*.dcm"):
        filename = dcm_path.split('/')[-1].replace('.dcm', '')
#         if int(instance_number) % 2 == 1:
#             continue
        dicom = dicomsdl.open(dcm_path)
        img = dicom.pixelData(storedvalue = True)

        if dicom['PixelRepresentation'] == 1:
            bit_shift = dicom['BitsAllocated'] - dicom['BitsStored']
            dtype = img.dtype
            img = (img << bit_shift).astype(dtype) >>  bit_shift
        img = img.astype(np.float32)

        intercept = dicom['RescaleIntercept']
        slope = dicom['RescaleSlope']
        if (slope is not None) & (intercept is not None):
            img = img * slope + intercept

        img = (img - img.min()) / (img.max() - img.min() + 1e-6)

        if dicom.PhotometricInterpretation == "MONOCHROME1":
            img = 1 - img
        img = (img*255.0).astype('uint8')
        save_path = f'input/sagittal_all_images/{study_id}___{series_id}___{filename}.png'
        paths.append(save_path)

        imgs[save_path] = img
        origin_paths.append(dcm_path)
        xyzs.append(dicom['ImagePositionPatient'])

    df = pd.DataFrame({
        'path': paths,
        'origin_path': origin_paths,
    })
    df[['x_pos', 'y_pos', 'z_pos']] = np.array(xyzs)
    df['study_id'] = study_id
    df['series_id'] = series_id
    df['instance_number'] = df['path'].apply(lambda x: int(x.split('___')[-1].replace('.png', '')))
    df = df.sort_values(['x_pos', 'instance_number'])
    df = df.drop_duplicates('x_pos')        
    path_list = df.path.values
    for path_n, path in enumerate(df.path):
        if path_n == 0:
            prev_path = df.path.values[0]
        else:
            prev_path = df.path.values[path_n-1]
        prev_im = imgs[prev_path]            
        
        im = imgs[path]
        if path_n == len(df)-1:
            next_path = df.path.values[-1]
        else:
            next_path = df.path.values[path_n+1]
        next_im = imgs[next_path]
  
        if not (prev_im.shape == im.shape == next_im.shape):

#             print(prev_im.shape, im.shape, next_im.shape)
            s = prev_im.shape
            im = cv2.resize(im, s)
            next_im = cv2.resize(next_im, s)

        image = np.array([prev_im, im, next_im]).transpose((1,2,0))
        cv2.imwrite(path, image)
    
    if verbose:
        rows = 7
        for n, k in enumerate(sorted(imgs.keys())):
            if n % rows == 0:
                fig = plt.figure(figsize=(20, 20))
            fig.add_subplot(1, rows, n%rows+1)

            im = imgs[k]
            plt.imshow(im, 'gray')

            if n % rows == rows-1:
                plt.show()        
        
    return df


# In[12]:


demo = False


# In[13]:


df = pd.read_csv('input/train_series_descriptions.csv')
if demo:
    df = df[df.study_id.isin(df.study_id.unique()[:40])]

axial_direction = pd.read_csv('input/axial_direction.csv')
dicom_dir = 'input/train_images'


# In[14]:


image_save_dir = 'input/axial_all_images'
os.makedirs(image_save_dir, exist_ok=True)

axial_df = df[df.series_description == 'Axial T2']
args = axial_df.drop_duplicates(['study_id', 'series_id'])[['study_id', 'series_id']].values


def read_axial_dicom(args, verbose=False):
    study_id, series_id = args
    imgs = {}
    origin_paths = []
    zs = []
    xyzs = []
    paths = []
    series_axial_direction = axial_direction[axial_direction.series_id==series_id]
    for dcm_path in glob(f"{dicom_dir}/{study_id}/{series_id}/*.dcm"):
        filename = dcm_path.split('/')[-1].replace('.dcm', '')
        dicom = dicomsdl.open(dcm_path)
        img = dicom.pixelData(storedvalue = True)

        if dicom['PixelRepresentation'] == 1:
            bit_shift = dicom['BitsAllocated'] - dicom['BitsStored']
            dtype = img.dtype
            img = (img << bit_shift).astype(dtype) >>  bit_shift
        img = img.astype(np.float32)

        intercept = dicom['RescaleIntercept']
        slope = dicom['RescaleSlope']
        if (slope is not None) & (intercept is not None):
            img = img * slope + intercept

        pos_z = dicom['ImagePositionPatient'][-1]
        
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)

        if dicom.PhotometricInterpretation == "MONOCHROME1":
            img = 1 - img
        img = (img*255.0).astype('uint8')
        save_path = f'input/axial_all_images/{study_id}___{series_id}___{filename}.png'
        paths.append(save_path)

        imgs[save_path] = img
        origin_paths.append(dcm_path)
        xyzs.append(dicom['ImagePositionPatient'])

    df = pd.DataFrame({
        'path': paths,
        'origin_path': origin_paths,
    })
    df[['x_pos', 'y_pos', 'z_pos']] = np.array(xyzs)
    df['study_id'] = study_id
    df['series_id'] = series_id
    df['instance_number'] = df['path'].apply(lambda x: int(x.split('___')[-1].replace('.png', '')))

    l = len(df)
    if len(df.merge(series_axial_direction[['instance_number', 'z']], on='instance_number')) == l: # If there are axial_direction info for all data
        df = df.merge(series_axial_direction[['instance_number', 'z']], on='instance_number')
        df = df.sort_values(['z', 'instance_number'])
    else:
        df = df.sort_values(['z_pos', 'instance_number'])
    path_list = df.path.values

    for path_n, path in enumerate(df.path):
        im = imgs[path]
        if path_n == 0:
            prev_path = df.path.values[0]
        else:
            prev_path = df.path.values[path_n-1]
        prev_im = imgs[prev_path]
        
        if path_n == len(df)-1:
            next_path = df.path.values[-1]
        else:
            next_path = df.path.values[path_n+1]
        next_im = imgs[next_path]
  
        if not (prev_im.shape == im.shape == next_im.shape):

            s = prev_im.shape
            im = cv2.resize(im, s)
            next_im = cv2.resize(next_im, s)

        image = np.array([prev_im, im, next_im]).transpose((1,2,0))
        cv2.imwrite(path, image)
        
    if verbose:
        rows = 7
        for n, k in enumerate(sorted(imgs.keys())):
            if n % rows == 0:
                fig = plt.figure(figsize=(20, 20))
            fig.add_subplot(1, rows, n%rows+1)

            im = imgs[k]
            plt.imshow(im, 'gray')

            if n % rows == rows-1:
                plt.show()        
        
    return df


# In[ ]:


from multiprocessing import Pool

p = Pool(processes=4)
results = []
with tqdm(total=len(args)) as pbar:
    for res in p.imap(read_axial_dicom, args):
        results.append(res)
        pbar.update(1)
p.close()

axial_df = pd.concat(results)
axial_df.to_csv('input/axial_df.csv', index=False)

image_save_dir = 'input/sagittal_all_images'
os.makedirs(image_save_dir, exist_ok=True)

sagittal_df = df[df.series_description != 'Axial T2']
args = sagittal_df.drop_duplicates(['study_id', 'series_id'])[['study_id', 'series_id']].values

from multiprocessing import Pool

p = Pool(processes=4)
results = []
with tqdm(total=len(args)) as pbar:
    for res in p.imap(read_sagittal_dicom, args):
        results.append(res)
        pbar.update(1)
p.close()

sagittal_df = pd.concat(results)

series_description_df = pd.read_csv(f'input/train_series_descriptions.csv')
sagittal_df = sagittal_df.merge(series_description_df, on=['study_id', 'series_id'])
sagittal_df.to_csv('input/sagittal_df.csv', index=False)

for study_id_n, (study_id, idf) in enumerate(tqdm(sagittal_df.groupby('study_id'))):
    t1 = idf[idf.series_description=='Sagittal T1'].sort_values('instance_number')
    if len(t1)!=0:
        t1 = t1[t1.series_id==t1.series_id.values[0]]
    t2 = idf[idf.series_description=='Sagittal T2/STIR'].sort_values('instance_number')
    if len(t2)!=0:
        t2 = t2[t2.series_id==t2.series_id.values[0]]
    if len(t1)==0:
        m = t2.instance_number.max()
        mi = t2.instance_number.min()
    elif (len(t2)==0):
        m = t1.instance_number.max()
        mi = t1.instance_number.min()
    else:
        m = max([t1.instance_number.max(), t2.instance_number.max()])
        mi = min([t1.instance_number.min(), t2.instance_number.min()])
    for n in range(mi, m+1):
        n1 = t1[t1.instance_number == n]
        n2 = t2[t2.instance_number == n]
        if len(n1) != 0:
            # print(n1.path.values[0])
            im1 = cv2.imread(n1.path.values[0])[:,:,1]
        else:
            im1 = None
        if len(n2) != 0:
            # print(n2.path.values[0])
            im2 = cv2.imread(n2.path.values[0])[:,:,1]
        else:
            im2 = None
        if im1 is None:
            # raise
            im1 = np.zeros(im2.shape)
        if im2 is None:
            # raise
            im2 = np.zeros(im1.shape)
        if im1.shape!=im2.shape:
            im1 = cv2.resize(im1, (im2.shape[1], im2.shape[0]))
        im = np.array([im1, im2, im1]).transpose((1,2,0))

        cv2.imwrite(f'input/sagittal_all_images/{study_id}___{n}.png', im)
