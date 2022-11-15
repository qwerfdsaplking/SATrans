import pandas as pd
import numpy as np
import joblib
import re
import random
from tqdm import tqdm
from utils import dump_pkl,load_pkl,dump_npy,load_npy
from utils import get_memory_info
import os
import gc
mini_freq=-1
random.seed(2020)
np.random.seed(2020)
data_path = '/home/featurize/data/sample_skeleton_{}.csv'
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


###################################

print('loading vocabulary...')
# vocabulary = load_pkl(enum_path)
vocabulary = joblib.load(enum_path)
uids = vocabulary['101']
uid_map = dict(
    zip(sorted(list(uids)), range(1, len(uids) + 1)))

for k,v in vocabulary.items():
    print(k,len(v))


min_uid=1
for col in acc_columns:
    get_memory_info()
    name = 'train'
    train_acc_feat = load_pkl('/home/featurize/data/history/%s_%s_unmapped.pkl' % (col, name))
    train_uids = train_acc_feat[0]
    train_fids = train_acc_feat[1]
    train_scores = train_acc_feat[2]
    # 过滤权重小的边，如何过滤？每个user限定5条边？还是统一用threshold去过滤

    name = 'test'
    test_acc_feat = load_pkl('/home/featurize/data/history/%s_%s_unmapped.pkl' % (col, name))
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
    print(min_fid,max_fid)

    for i,x in tqdm(enumerate(train_fids)):
        train_fids[i]=x-min_fid+1
    for i,x in tqdm(enumerate(test_fids)):
        test_fids[i]=x-min_fid+1
    print(max_fid-min_fid+1,len(set(fids)))

    dump_pkl([train_uids, train_fids, train_scores], '/home/featurize/data/history/%s_%s_mapped.pkl' % (col, name))
    dump_pkl([test_uids, test_fids, test_scores], '/home/featurize/data/history/%s_%s_mapped.pkl' % (col, name))
    del train_acc_feat,test_acc_feat,train_uids,train_fids,train_scores,test_uids,test_fids,test_scores
    gc.collect()


class process(object):
    def __init__(self):
        pass








    def process_history(self):
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
                dump_pkl(acc_dic[col],'/home/featurize/data/history/%s_%s_unmapped.pkl' %(col,name))







            #print(len(vocabulary[item_col[i]]),len(acc_set[col]))
        #for i in range(4):
        #    print(len(vocabulary[item_col[i]]),len(acc_set[acc_columns[i]]),
        #          min(vocabulary[item_col[i]]),min(acc_set[acc_columns[i]]),
        #          max(vocabulary[item_col[i]]),max(acc_set[acc_columns[i]]))





    def process_train(self):
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
                if '101' not in key:
                    print(key)
                value = kv[range(1, len(kv), 3)]
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
        #dump_pkl(acc_dic,'/home/featurize/data/train_history_edges_unmapped.pkl')
        name='test'
        acc_dic = load_pkl('/home/featurize/data/%s_history_edges_unmapped.pkl'%name)

        for col in acc_columns:
            dump_pkl(acc_dic[col], '/home/featurize/data/history/%s_%s_unmapped.pkl' % (col, name))

        item_set=dict()
        item_col = ['206','207','210','216']


        for i,col in enumerate(acc_columns):
            a=pd.Series([float(x) for x in acc_dic[col][2]])
            print(a.describe())
            acc_set[col] = set(acc_dic[col][1])
            item_set[item_col[i]]



        print('join feats...')
        c = 0
        vocabulary = dict(zip(use_columns, [{}  for _ in range(len(use_columns))]))
        with open(data_path.format('train') + '.tmp', 'w') as fw:
            fw.write('click,purchase,' + ','.join(use_columns) + '\n')
            with open(data_path.format('train'), 'r') as fr:
                for line in fr:
                    line_list = line.strip().split(',')
                    if line_list[1] == '0' and line_list[2] == '1':
                        continue
                    kv = np.array(re.split('\x01|\x02|\x03', line_list[5]))
                    key = kv[range(0, len(kv), 3)]
                    value = kv[range(1, len(kv), 3)]
                    feat_dict = dict(zip(key, value))
                    feat_dict.update(common_feat_dict[line_list[3]])
                    feats = line_list[1:3]
                    for k in use_columns:
                        feats.append(feat_dict.get(k, '0'))
                    fw.write(','.join(feats) + '\n')
                    for k, v in feat_dict.items():
                        if k in use_columns:
                            if v in vocabulary[k]:
                                vocabulary[k][v] += 1
                            else:
                                vocabulary[k][v] = 1
                    c += 1
                    if c % 100000 == 0:
                        print(c)



        print('before filter low freq:')
        for k, v in vocabulary.items():
            print(k + ':' + str(len(v)))
        new_vocabulary = dict(
            zip(use_columns, [set() for _ in range(len(use_columns))]))
        for k, v in vocabulary.items():
            for k1, v1 in v.items():
                if v1 > mini_freq:
                    new_vocabulary[k].add(k1)
        vocabulary = new_vocabulary
        print('after filter low freq:')
        for k, v in vocabulary.items():
            print(k + ':' + str(len(v)))
            vocabulary[k] = sorted(list(v))
        dump_pkl(vocabulary, enum_path)#, compress=3)

        print('encode feats...')
        vocabulary = load_pkl(enum_path)
        feat_map = {}
        for feat in use_columns:
            feat_map[feat] = dict(
                zip(sorted(list(vocabulary[feat])), range(1, len(vocabulary[feat]) + 1)))
        c = 0
        with open(write_path + '.train', 'w') as fw1:
            with open(write_path + '.dev', 'w') as fw2:
                with open(write_path+'.all','w') as fw3:
                    fw1.write('click,purchase,' + ','.join(use_columns) + '\n')
                    fw2.write('click,purchase,' + ','.join(use_columns) + '\n')
                    fw3.write('click,purchase,' + ','.join(use_columns) + '\n')

                    with open(data_path.format('train') + '.tmp', 'r') as fr:
                        fr.readline()  # remove header
                        for line in fr:
                            line_list = line.strip().split(',')
                            new_line = line_list[:2]
                            for value, feat in zip(line_list[2:], use_columns):
                                new_line.append(
                                    str(feat_map[feat].get(value, '0')))
                            if random.random() >= 0.9:
                                fw2.write(','.join(new_line) + '\n')
                            else:
                                fw1.write(','.join(new_line) + '\n')
                            fw3.write(','.join(new_line) + '\n')
                            c += 1
                            if c % 100000 == 0:
                                print(c)

    def process_test(self):
        c = 0
        common_feat_dict = {}
        acc_dic = dict()
        acc_set = dict()
        for col in acc_columns:
            acc_dic[col] = [[],[],[]]
            acc_set[col] = set()
        with open(common_feat_path.format('test'), 'r') as fr:
            for line in tqdm(fr):
                line_list = line.strip().split(',')
                kv = np.array(re.split('\x01|\x02|\x03', line_list[2]))
                key = kv[range(0, len(kv), 3)]
                value = kv[range(1, len(kv), 3)]
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
        dump_pkl(acc_dic,'/home/featurize/data/test_history_edges_unmapped.pkl')


        print('join feats...')
        c = 0
        with open(data_path.format('test') + '.tmp', 'w') as fw:
            fw.write('click,purchase,' + ','.join(use_columns) + '\n')
            with open(data_path.format('test'), 'r') as fr:
                for line in fr:
                    line_list = line.strip().split(',')
                    if line_list[1] == '0' and line_list[2] == '1':
                        continue
                    kv = np.array(re.split('\x01|\x02|\x03', line_list[5]))
                    key = kv[range(0, len(kv), 3)]
                    value = kv[range(1, len(kv), 3)]
                    feat_dict = dict(zip(key, value))
                    feat_dict.update(common_feat_dict[line_list[3]])
                    feats = line_list[1:3]
                    for k in use_columns:
                        feats.append(str(feat_dict.get(k, '0')))
                    fw.write(','.join(feats) + '\n')
                    c += 1
                    if c % 100000 == 0:
                        print(c)

        print('encode feats...')
        vocabulary = load_pkl(enum_path)
        feat_map = {}
        for feat in use_columns:
            feat_map[feat] = dict(
                zip(vocabulary[feat], range(1, len(vocabulary[feat]) + 1)))
        c = 0
        with open(write_path + '.test', 'w') as fw:
            fw.write('click,purchase,' + ','.join(use_columns) + '\n')
            with open(data_path.format('test') + '.tmp', 'r') as fr:
                fr.readline()  # remove header
                for line in fr:
                    line_list = line.strip().split(',')
                    new_line = line_list[:2]
                    for value, feat in zip(line_list[2:], use_columns):
                        new_line.append(str(feat_map[feat].get(value, '0')))
                    fw.write(','.join(new_line) + '\n')
                    c += 1
                    if c % 100000 == 0:
                        print(c)




if __name__ == "__main__":
    print('---')
    pros = process()
    #pros.process_train()
    #pros.process_test()
    #pros.process_history()
