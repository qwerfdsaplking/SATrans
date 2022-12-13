import pandas as pd

import numpy as np
from datetime import datetime
data_path  = '../../data/'

if __name__=='__main__':


    logs = pd.read_csv(f'{data_path}raw_sample.csv')
    print('logs load finished')
    times = logs['time_stamp'].map(lambda x: datetime.fromtimestamp(x))
    gaps = [('2017-05-%s 00:00:00'%('%02d'%i), '2017-05-%s 23:59:59'%('%02d'%i)) for i in range(6,14)]

    item_df = pd.read_csv(f'{data_path}ad_feature.csv')

    user_df = pd.read_csv(f'{data_path}user_profile.csv')

    logs.columns = ['user_id', 'time_stamp', 'adgroup_id', 'pid', 'nonclk', 'clk']
    user_df.columns = ['userid', 'cms_segid', 'cms_group_id', 'final_gender_code', 'age_level','pvalue_level', 'shopping_level', 'occupation', 'new_user_class_level']
    data_error = logs.join(user_df,on='user_id',how='left')

    print('join')
    data = logs.merge(user_df,on='user_id',how='left')
    data = data.merge(item_df,on='adgroup_id',how='left')

    data[['cms_segid', 'cms_group_id','age_level','occupation']]+=1
    data = data.fillna(value=0)

    pid_map={'430548_1007':0, '430539_1007':1}
    data['pid'] = data['pid'].map(pid_map)

    def re_index(data,col):
        vals= data[col].unique().tolist()
        vals = sorted(vals)
        vals_map = dict(zip(vals,range(len(vals))))
        data[col] = data[col].map(vals_map)
        return data
    #data = re_index(data,'brand')
    #data = re_index(data,'cate_id')
    def save_df2h5(hdf5_path, df):
        import h5py
        print('save %s ...' % hdf5_path)
        f = h5py.File(hdf5_path, 'w')
        for col in df.columns:
            if col not in ['price']:
                f[col]=df[col].values.astype('int')
            else:
                f[col] = df[col].values.astype('float')
        f.close()

    save_df2h5(f'{data_path}alimama.h5',data)