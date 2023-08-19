import pandas as pd
import numpy as np
import joblib
import re
import random
from tqdm import tqdm
from utils import dump_pkl,load_pkl,dump_npy,load_npy
from utils import *
import os
import gc
mini_freq=-1
random.seed(2020)
np.random.seed(2020)
data_path = '/home/featurize/data/sample_skeleton_{}.csv'
norm_data_path = '/home/featurize/data/data_norm_{}.csv'

common_feat_path = '/home/featurize/data/common_features_{}.csv'
enum_path = '/home/featurize/data/ctrcvr_enum.pkl'
write_path = '/home/featurize/data/ctr_cvr'
acc_uf_edge_path = '/home/featurize/data/uf_edge_path.csv'
use_columns = [
    '101',
    '121',
    '122',
    '124',
    '125',
    '126',
    '127',
    '128',
    '129',
    '205',
    '206',
    '207',
    '210',
    '216',
    '508',
    '509',
    '702',
    '853',
    '301'] #+
acc_columns = ['109_14','110_14','150_14','127_14']

h5_path = '/home/featurize/work/alicpp.h5'

###################################



def map_history_data():
    #统一编码从1开始
    min_uid=1
    for col in ['109_14','110_14','150_14','127_14']:
        get_memory_info()
        name = 'train'
        train_acc_feat = load_pkl('/home/featurize/work/history/%s_%s_unmapped.pkl' % (col, name))
        train_uids = train_acc_feat[0]
        train_fids = train_acc_feat[1]
        train_scores = train_acc_feat[2]

        name = 'test'
        test_acc_feat = load_pkl('/home/featurize/work/history/%s_%s_unmapped.pkl' % (col, name))
        test_uids = test_acc_feat[0]
        test_fids = test_acc_feat[1]
        test_scores = test_acc_feat[2]

        for i,x in tqdm(enumerate(train_uids)):
            train_uids[i]=x-min_uid+1
        for i,x in tqdm(enumerate(test_uids)):
            test_uids[i]=x-min_uid+1
        fids = train_fids+test_fids
        min_fid = min(fids)
        max_fid = max(fids)
        print(col, min_fid,max_fid)
        #109_14 444862 457385
        #110_14 457386 3438657
        #150_14 3864891 3964446
        #127_14 3438783 3864884

        for i,x in tqdm(enumerate(train_fids)):
            train_fids[i]=x-min_fid+1
        for i,x in tqdm(enumerate(test_fids)):
            test_fids[i]=x-min_fid+1
        #print(max_fid-min_fid+1,len(set(fids)))

        #col = col.split('_')[0]
        col = col.replace('_', '')
        data =np.array([train_uids, train_fids, train_scores],dtype=np.float)
        save_h5(h5_path, '101_%s/train' %col, data)
        data =np.array([test_uids, test_fids, test_scores],dtype=np.float)
        save_h5(h5_path, '101_%s/test' %col, data)

        del data,train_acc_feat,test_acc_feat,train_uids,train_fids,train_scores,test_uids,test_fids,test_scores
        gc.collect()


def get_topk_edges(data):
    edges = pd.DataFrame(data.T, columns=['uid', 'fid', 'score'])
    print('before drop duplicates',edges.shape[0])
    edges = edges.drop_duplicates()
    print('after drop duplicates',edges.shape[0])
    #func = lambda x:x.sort_values(['score'],ascending=False)[:20]
    #edges20 = edges.groupby('uid').apply(func)
    print('all',edges.shape,edges['score'].max())
    edges = edges.sort_values(by=['uid', 'score'], ascending=(True, False))
    edges20 = edges.groupby('uid').head(20)
    print('top20', edges20.shape)
    edges10 = edges.groupby('uid').head(10)
    print('top10', edges10.shape)
    edges5 = edges.groupby('uid').head(5)
    print('top5', edges5.shape,edges5['score'].max())
    return edges5.values,edges10.values,edges20.values

def merge_train_test_history_edges(train_edges,test_edges):
    print('train&test shape',train_edges.shape,test_edges.shape)
    train_uids_set = set(train_edges.T[0].tolist())
    test_uids = test_edges.T[0].tolist()
    idx = [False if x in train_uids_set else True for x in test_uids]
    test_edges_filtered = test_edges[idx]
    print('test&filtered shape',test_edges.shape,test_edges_filtered.shape)
    merged_edges = np.concatenate([train_edges,test_edges_filtered],axis=0)
    print('merged shape',merged_edges.shape)
    return merged_edges

def save_topk_edges():
    for col in acc_columns:
        print(col)
        for name in ['train','test']:
            #col=col.split('_')[0]
            col = col.replace('_','')
            data = load_h5(h5_path,'101_%s/%s'%(col,name))
            edges5,edges10,edges20 = get_topk_edges(data)
            save_h5(h5_path,'101_%s/top%s_%s'%(col,5,name),edges5)
            save_h5(h5_path,'101_%s/top%s_%s'%(col,10,name),edges10)
            save_h5(h5_path,'101_%s/top%s_%s'%(col,20,name),edges20)

    for col in acc_columns:
        for k in [5,10,20]:
            #col=col.split('_')[0]
            col = col.replace('_','')
            train_edges = load_h5(h5_path,'101_%s/top%s_train'%(col,k))
            test_edges = load_h5(h5_path,'101_%s/top%s_test'%(col,k))
            merged_edges = merge_train_test_history_edges(train_edges,test_edges)
            save_h5(h5_path,'101_%s/top%s_merged'%(col,k),merged_edges)





#映射带特征的点击数据
def normalize_train_and_test():

    import pandas as pd
    import numpy as np
    data_path = '/home/featurize/data/sample_skeleton_{}.csv'
    norm_data_path = '/home/featurize/data/data_norm_{}.csv'
    id_feat_edges_path  = '/home/featurize/data/id_feat_edges.pkl'

    min_v = {'click':1,'purchase':1,'101': 1, '121': 3438658, '122': 3438755, '124': 3438768, '125': 3438770, '126': 3438777, '127': 3438780, '128': 3864885, '129': 3864887, '205': 3964447, '206': 8313062, '207': 8322055, '210': 9017179, '216': 9116785, '508': 9351668, '509': 9359853, '702': 9832207, '853': 10000020, '301': 9351665}
    min_v = pd.Series(min_v)
    #norm_data_path = '../data_norm_{}.csv'
    def norm_df(path,out_path):
        df = pd.read_csv(path,dtype=np.int32)
        print(df.shape)
        df -= (min_v-1)
        df[df<0]=0
        df =df.astype(np.int32)
        print(df.head(10))
        df.to_csv(out_path,index=False)
        return df
    train_df = norm_df(data_path.format('train') + '.tmp',
            norm_data_path.format('train'))
    test_df = norm_df(data_path.format('test') + '.tmp',
            norm_data_path.format('test'))
    #train_max = train_df.max()
    #test_max = test_df.max()
    #data_max = {}
    #for key in train_df.columns:
    #    data_max[key] = max(train_max[key],test_max[key])
    data_max = {'101': 444861, '121': 97, '122': 13, '124': 2, '125': 7, '126': 3, '127': 3, '128': 2, '129': 4, '205': 4348615, '206': 8993, '207': 695124, '210': 99606, '216': 234880, '508': 8185, '509': 472354, '702': 167813, '853': 91358, '301': 3}

    for col in train_df.columns:
        print(train_df[col].unique().shape[0])
    for col in test_df.columns:
        print(test_df[col].unique().shape[0])


    edge_dict = {}
    user_cols = ['121', '122', '124', '125', '126', '127', '128', '129']
    item_cols = ['206', '207', '210', '216']
    for col in user_cols:
        feat_df = pd.concat([train_df[['101',col]],test_df[['101',col]]],axis=0)
        user_feat = feat_df.drop_duplicates().values.T
        print(user_feat.shape[1])
        edge_dict['101_'+col] = user_feat


    for col in item_cols:
        feat_df = pd.concat([train_df[['205', col]],test_df[['205', col]]],axis=0)
        feat_df = feat_df[(feat_df['205']!=0)]
        feat_df = feat_df[(feat_df[col]!=0)]
        item_feat = feat_df.drop_duplicates().values.T
        print(item_feat.shape[1])
        edge_dict['205_'+col] = item_feat

    for key in edge_dict:
        save_h5(h5_path,'%s/merged'%(key), edge_dict[key])




def generate_topk_history_features():
    for name in ['train','test']:
        data_all = pd.read_csv('/home/featurize/data/data_norm_%s.csv' %name, dtype=np.int32)
        history_dfs = []
        for k in [5]:
            for col in ['109_14','110_14','150_14','127_14']:
                col = col.replace('_', '')
                edges = load_h5(h5_path, '101_%s/top%s_%s' % (col, k,name))
                print(edges.shape)
                df = pd.DataFrame(edges[:,:2],columns=['101',col],dtype=np.int)
                df = df.groupby('101').aggregate(lambda x: tuple(list(x)+[0]*(k-len(x)))).reset_index()
                history_dfs.append(df.set_index(['101']))
                print(df.shape)

        history_df = pd.concat(history_dfs, axis=1).reset_index()
        for col in history_df.columns[1:]:
            history_df[col] = history_df[col].apply(lambda x: tuple([0]*k if type(x) != tuple else x))
        print(history_df.shape)

        data_all = data_all.merge(history_df, on='101', how='left')
        print(data_all.isna().sum())
        for col in ['10914','11014','15014','12714']:
            data_all[col] = data_all[col].apply(lambda x: tuple([0]*k if type(x) != tuple else x))



        for col in data_all.columns:
            if col in ['10914','11014','15014','12714']:
                vs = np.array(list(data_all[col]), dtype=np.int32)
            else:
                vs = data_all[col].values
            save_h5(h5_path,f'ctr_{name}/{col}', vs)
        #data_all.to_csv('/home/featurize/data/data_norm_%s_with_histfeats.csv' %name)




        #test_edges = load_h5(h5_path, '101_%s/top%s_test' % (col, k))





def process_history():
    for name in ['train', 'test']:
        file_path = '/home/featurize/data/%s_history_edges_unmapped.pkl' %name
        if not os.path.exists(file_path):
            c = 0
            common_feat_dict = {}
            acc_dic = dict()
            acc_set = dict()
            for col in acc_columns:
                acc_dic[col] = [[],[],[]]
                acc_set[col] = set()
            with open(common_feat_path.format('train'), 'r') as fr:
                for line in tqdm(fr):
                    line_list = line.strip().split(',')
                    kv = np.array(re.split('\x01|\x02|\x03', line_list[2]))
                    key = kv[range(0, len(kv), 3)]
                    value = kv[range(1, len(kv), 3)]
                    if len(value)!=len(set(value)):
                        print(len(value),len(set(value)),value)

                    score = kv[range(2, len(kv), 3)]
                    kvs_zip = list(zip(key, value,score))
                    feat_dict = dict(zip(key, value))
                    common_feat_dict[line_list[0]] = feat_dict

                    if '101' in feat_dict:#有的特征没有uid?
                        uid = feat_dict['101']
                        for col in acc_columns:
                            acc_dic[col][0] += [int(uid) for x in kvs_zip if x[0]==col]
                            acc_dic[col][1] += [int(x[1]) for x in kvs_zip if x[0]==col]
                            acc_dic[col][2] += [float(x[2]) for x in kvs_zip if x[0] == col]
                    c += 1
                    if c % 10000 == 0:
                        print('')
                        for col in acc_columns:
                            print(len(acc_dic[col][0]))
                        print(get_memory_info())
        #dump_pkl(acc_dic,file_path)
        for col in acc_columns:
            a=pd.DataFrame(np.array(acc_dic[col][:2]).T,columns=['uid','fid'])

            dump_pkl(acc_dic[col],'/home/featurize/data/history/%s_%s_unmapped.pkl' %(col,name))






if __name__ == "__main__":
    print('---')
    #pros = process()

    map_history_data()
    save_topk_edges()
    normalize_train_and_test()
    generate_topk_history_features()
    #normalize_train_and_test()
    #pros.process_train()
    #pros.process_test()
    #pros.process_history()
