import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
import gc
import torch
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import *
plt.style.use('ggplot')
from sklearn.preprocessing import LabelEncoder

import warnings as w
w.filterwarnings(action='ignore')
pd.set_option('display.max_columns',None)

dtype={'id': np.dtype(int),
    'click': np.dtype(int),
    'hour': np.str,
    'C1': np.dtype(str),
    'banner_pos': np.dtype(str),
    'site_id': np.dtype(str),
    'site_domain': np.dtype(str),
    'site_category': np.dtype(str),
    'app_id': np.dtype(str),
    'app_domain': np.dtype(str),
    'app_category': np.dtype(str),
    'device_id': np.dtype(str),
    'device_ip': np.dtype(str),
    'device_model': np.dtype(str),
    'device_type': np.dtype(str),
    'device_conn_type': np.dtype(str),
    'C14': np.dtype(str),
    'C15': np.dtype(str),
    'C16': np.dtype(str),
    'C17': np.dtype(str),
    'C18': np.dtype(str),
    'C19': np.dtype(str),
    'C20': np.dtype(str),
    'C21':np.dtype(str)
      }



#train = pd.read_csv("~/data/train.gz", parse_dates=['hour'], date_parser=parse_date, dtype=dtype, skiprows=skip_values)
data = pd.read_csv('~/data/train.gz')
train=data[data['hour']<=14102923]
test=data[data['hour']>14102923]

print('Train dataset:',train.shape)
print('Test dataset:',test.shape)

feat_cols = ['C1', 'banner_pos', 'site_id', 'site_domain',
       'site_category', 'app_id', 'app_domain', 'app_category', 'device_id',
       'device_ip', 'device_model', 'device_type', 'device_conn_type', 'C14',
       'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']

for col in feat_cols:

    #data[col] = LabelEncoder().fit_transform(data[col])
    print(col,data[col].min(),data[col].max())

def save_feat2hdf5(hdf5_path, data_dict):
    import h5py
    print('Save %s ...' % hdf5_path)
    f = h5py.File(hdf5_path, 'w')
    for k,v in data_dict.items():
        f[k] = v
    f.close()

def save_df2h5(hdf5_path, data_df):
    import h5py
    print('Save %s ...' % hdf5_path)
    f = h5py.File(hdf5_path, 'w')
    for col in data_df.columns:
        print(col)
        f[col]=data_df[col].values.astype('int32')
    f.close()

def loadh52df(hdf5_path,columns):
    import h5py
    print('load %s ...' % hdf5_path)
    f = h5py.File(hdf5_path, 'r')
    dic = {}
    for col in columns:
        dic[col] = f[col][:]
    f.close()
    return pd.DataFrame(dic,columns=columns)
save_df2h5('/home/featurize/work/data/avazu.h5',data)