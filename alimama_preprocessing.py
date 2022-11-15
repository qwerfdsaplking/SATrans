import pandas as pd

import numpy as np
from datetime import datetime
data_path  = '~/data/'

if __name__=='__main__':


    logs = pd.read_csv(f'{data_path}raw_sample.csv')
    print('logs load finished')
    times = logs['time_stamp'].map(lambda x: datetime.fromtimestamp(x))
    gaps = [('2017-05-%s 00:00:00'%('%02d'%i), '2017-05-%s 23:59:59'%('%02d'%i)) for i in range(6,14)]

    item_df = pd.read_csv(f'{data_path}ad_feature.csv')

    user_df = pd.read_csv(f'{data_path}user_profile.csv')

    logs.columns = ['user_id', 'time_stamp', 'adgroup_id', 'pid', 'nonclk', 'clk']
    user_df.columns = ['userid', 'cms_segid', 'cms_group_id', 'final_gender_code', 'age_level','pvalue_level', 'shopping_level', 'occupation', 'new_user_class_level']

    data = logs.join(user_df,on='user_id',how='left')
    data = data.merge(item_df,on='adgroup_id',how='left')

    data = data.fillna(value=0)

    pid_map={'430548_1007':1, '430539_1007':2}
    data['pid'] = data['pid'].map(pid_map)

    def re_index(data,col):
        vals= data[col].unique().tolist()
        vals = sorted(vals)
        vals_map = dict(zip(vals,range(len(vals))))
        data[col] = data[col].map(vals_map)
        return data
    #data = re_index(data,'brand')
    #data = re_index(data,'cate_id')

    data.to_csv(f'{data_path}alimama.csv', index=False)