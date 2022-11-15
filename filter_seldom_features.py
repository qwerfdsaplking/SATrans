import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from deepctr_torch.models import *
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names,VarLenSparseFeat
from argparse import ArgumentParser
import warnings
warnings.filterwarnings('ignore')
from utils import *
from torch import nn
from models.star_trans import *
from models.dcn import *
from models.deepfm import *
from models.autoint import *

class Star_Net_old(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_dim, feat_dim_dict, feat_list, num_domains):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_domains = num_domains
        self.emb_dict = nn.ModuleDict()
        for k, num_nodes in feat_dim_dict.items():
            self.emb_dict[k] = nn.Embedding(num_nodes+1, embedding_dim, padding_idx=0)

        self.linear1 = nn.Linear(embedding_dim*len(feat_list), hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)

        input_dim = embedding_dim*len(feat_list)

        self.drop = nn.Dropout()
        self.act_fn = nn.ReLU()

        self.domain_kernel_1 = nn.ParameterList([nn.Parameter(torch.randn(input_dim, hidden_dim)) for i in range(num_domains)])
        self.domain_bias_1 = nn.ParameterList([nn.Parameter(torch.randn(hidden_dim, 1)) for i in range(num_domains)])
        self.domain_kernel_2 = nn.ParameterList([nn.Parameter(torch.randn(input_dim, hidden_dim)) for i in range(num_domains)])
        self.domain_bias_2 = nn.ParameterList([nn.Parameter(torch.randn(hidden_dim, 1)) for i in range(num_domains)])

        self.shared_kernel_1 = nn.Parameter(torch.randn(input_dim, hidden_dim))
        self.shared_bias_1 = nn.Parameter(torch.randn(hidden_dim, 1))
        self.shared_kernel_2 = nn.Parameter(torch.randn(input_dim, hidden_dim))
        self.shared_bias_2 = nn.Parameter(torch.randn(hidden_dim, 1))

        self.domain_linear = nn.Linear(embedding_dim,1)

    def forward(self, batch, batch_size):
        out_dict = {}

        for k, v in batch.items():
            out_dict[k] = self.drop(self.emb_dict[k](v).relu_())
        domain_id = batch['pid']

        out_list = [out_dict[k] for k in self.feature_list]
        out = torch.cat(out_list, dim=-1)

        logits = torch.zeros(out.shape[0])

        for i in range(self.num_domains):
            out_domain = out[domain_id == i+1]
            domain_kernel_1 = self.shared_kernel_1 * self.domain_kernel_1[i]
            domain_bias_1 = self.shared_bias_1 + self.domain_bias_1[i]
            out_domain = self.drop(self.act_fn(out_domain@domain_kernel_1 + domain_bias_1))

            domain_kernel_2 = self.shared_kernel_2 * self.domain_kernel_2[i]
            domain_bias_2 = self.shared_bias_2 + self.domain_bias_2[i]
            out_domain = out_domain@domain_kernel_2 + domain_bias_2
            logits[domain_id == i+1] += out_domain

        domain_logits = self.domain_linear(out_dict['pid'])

        return logits + domain_logits



def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_name', type=str, default='alimama')
    parser.add_argument('--model_name', type=str, default='Starv2_Trans')
    parser.add_argument('--seed', type=str, default=1024)
    parser.add_argument('--merge', type=str, default='no')
    parser.add_argument('--num_query_bases', type=int, default=3)
    parser.add_argument('--share_domain_dnn_across_layers', type=boolean_string,default=False )
    parser.add_argument('--domain_col',type=str,default='None')
    parser.add_argument('--embedding_dim',type=int,default=32)
    parser.add_argument('--att_layer_num',type=int,default=0)
    parser.add_argument('--domain_att_layer_num',type=int,default=0)
    parser.add_argument('--att_layer_type',type=str,default='trans')
    parser.add_argument('--att_head_num',type=int,default=2)
    parser.add_argument('--flag',type=str,default='None')
    #{att_layer_num}_{domain_att_layer_num}


    args = parser.parse_args()
    # set_model_scale(args.model_scale, args)
    return args


def embedding_initialization(model, embedding_path):
    print('embedding initialization')
    init_embedding_dict = load_pkl(embedding_path)
    for key in model.embedding_dict.keys():
        if key in init_embedding_dict:
            assert model.embedding_dict[key].weight.shape == init_embedding_dict[key].shape
            model.embedding_dict[key].weight = nn.Parameter(init_embedding_dict[key])

def generate_topk_history_features():
    h5_path = '/home/featurize/work/alicpp.h5'
    res = []
    for name in ['train','test']:
        data_all = pd.read_csv('/home/featurize/data/data_norm_%s.csv' %name, dtype=np.int32)
        history_dfs = []
        for k in [10]:
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
        res.append(data_all)
    return res

def tmp_reduce(path,cols):
    print(cols)
    h5_path = '/home/featurize/work/alicpp.h5'
    f = h5py.File(h5_path, 'a')
    group = f[path]
    data_dict={}
    for k in cols:
        print(k)
        if k in ['10914', '11014', '15014', '12714']:
            #group[k+'_3'] = group[k][:][:,:3]
            #group[k + '_5'] = group[k][:][:, :5]
            group[k+'_1'] = group[k][:][:,:1].reshape(-1)

    return data_dict



def get_ctr_df(path,cols,k=3):
    print(cols)
    h5_path = '/home/featurize/work/alicpp.h5'
    f = h5py.File(h5_path, 'r')
    group = f[path]
    data_dict={}
    for key in cols:
        #print('loading key:',key)
        if key in ['10914', '11014', '15014', '12714']:
            new_key = key+'_'+str(int(k))
        else:
            new_key = key
        data_dict[key] = group[new_key][:]
    return data_dict

def get_sub_context(data_dict,cid):
    idx = data_dict['301']==cid
    sub_data_dict={}
    for key in data_dict:
        sub_data_dict[key]=data_dict[key][idx]
    print('all num',idx.shape[0],f'sub {cid} num',idx.sum())
    return sub_data_dict



data_name = 'alimama'
#data_name = 'alicpp'


hidden_dim = 80
seed = 1024
batch_size=4096*2
if __name__== '__main__':
    print('starting...4++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    args = parse_args()
    model_name=args.model_name
    data_name=args.data_name
    seed=args.seed
    merge=args.merge
    domain_col  = args.domain_col
    embedding_dim=args.embedding_dim
    att_head_num = args.att_head_num
    att_layer_type = args.att_layer_type
    att_layer_num = args.att_layer_num
    domain_att_layer_num = args.domain_att_layer_num
    flag=args.flag
    print(args)


    default_domain_col_dict = {'alicpp':'301','alimama':'pid','ijcai18':'user_gender_id'}
    if domain_col == 'None':
        domain_col = default_domain_col_dict[data_name]


    #data_name_list = [ 'alicpp']#,'alicpp']#'alimama']'Star_Net',
    #model_name_list = ['Star_Trans','Star_Trans_w_Out']#'DeepFM']#,'DCN','DeepFM','AutoInt','NFM']#STAR_Trans
    #model_name_list = ['Star_Net_unorm','Star_Net','WDL','AutoInt','DCN']#'DeepFM']#,'DCN','DeepFM','AutoInt','NFM']#STAR_Trans
    #print(model_name_list)
    #model_name_list = ['xDeepFM','DCN','DeepFM','AutoInt','NFM']
    if data_name == 'alicpp':
        labels = ['click']
        uid = ['101']
        tid = ['205']
        sparse_features = ['101', '121', '122', '124', '125', '126', '127', '128', '129', '205', '206', '207', '210', '216',
                           '508', '509', '702', '853', '301']
        #var_features = ['10914', '11014', '15014', '12714']
        var_features = []
        dense_features = []
        topk=3
        train_all = get_ctr_df('ctr_train',labels+sparse_features+var_features,k=topk)
        print('load train finish')
        test_all = get_ctr_df('ctr_test',labels+sparse_features+var_features,k=topk)
        print('load test finish')

        if train_all[domain_col].min()==0:
            train_all[domain_col]+=1
            test_all[domain_col]+=1
        print(pd.Series(train_all[domain_col]).value_counts())

        def get_domain_feat(data,domain_col,dids_map=None):
            columns = train_all.keys() if isinstance(train_all,dict) else train_all.columns.tolist()
            if domain_col not in columns:
                domain_cols = domain_col.split('|')
                domain_feats = pd.DataFrame(np.concatenate([train_all[col].reshape(-1,1) for col in domain_cols],axis=1),columns=domain_cols) if isinstance(train_all,dict) else data[domain_cols]
                domain_ids = domain_feats.apply(lambda x:tuple(x), axis=1)
                dids_set = set(domain_ids.unique().tolist())
                if not dids_map:
                    dids_map = dict(zip(dids_set,range(len(dids_set))))
                domain_ids = domain_ids.map(lambda x:dids_map[x]).value
            else:
                domain_cols = [domain_col]
                domain_ids = data[domain_col]
            data[domain_col] = domain_ids
            return domain_cols, dids_map

        #domain_cols, did_map = get_domain_feat(train_all, domain_col, None)
        #domain_cols, did_map = get_domain_feat(test_all, domain_col, did_map)
        def filter_seldom_feats(data,voca_dic=None):
            mini_freq=10

            new_voca_dic=dict()

            for col in data.keys():
                if col in labels+uid+tid:
                    continue

                val_col = pd.Series(data[col])
                if not voca_dic:
                    vals_cnts = val_col.value_counts()
                    valid_vals = vals_cnts[vals_cnts>=mini_freq].index.values
                    print(col,vals_cnts.shape,valid_vals.shape,val_col.min())
                    valid_vals = set(valid_vals.tolist())
                    new_voca_dic[col]=valid_vals
                else:
                    valid_vals=voca_dic[col]
                val_col = val_col.map(lambda x:x if x in valid_vals else int(0)).astype(np.int32)
                data[col]=val_col.values
            return new_voca_dic

        voca_dic = filter_seldom_feats(train_all)
        filter_seldom_feats(test_all, voca_dic)



        def save_ctr_df(path, data,cols):
            print(cols)
            h5_path = '/home/featurize/work/alicpp.h5'
            f = h5py.File(h5_path, 'a')
            if path in f.keys():
                del f[path]
            for key in cols:

                f[path+'/'+key] = data[key]

        save_ctr_df('ctr_train_filtered',train_all,train_all.keys())
        save_ctr_df('ctr_test_filtered',test_all,test_all.keys())




        num_domains = pd.Series(train_all[domain_col]).nunique()

    elif data_name=='alimama':
        labels = ['clk']
        sparse_features = ['user_id','adgroup_id', 'pid','userid', 'cms_segid', 'cms_group_id', 'final_gender_code', 'age_level', 'pvalue_level', 'shopping_level', 'occupation', 'new_user_class_level', 'cate_id', 'campaign_id', 'customer', 'brand']
        var_features = []
        dense_features = ['price']
        data = pd.read_csv('~/data/alimama.csv')

        #domain_col = 'pid'
        if data[domain_col].min()==0:
            data[domain_col]+=1
        print(data[domain_col].value_counts())
        mms = MinMaxScaler(feature_range=(0, 1))
        data[dense_features] = mms.fit_transform(data[dense_features])
        import time
        split_time_stamp = '2017-05-12 00:00:00'
        ts = time.mktime(time.strptime(split_time_stamp, "%Y-%m-%d %H:%M:%S"))
        train_all = data[data['time_stamp'] < ts]
        test_all = data[data['time_stamp'] >= ts]
        num_domains = 2
    elif data_name =='ijcai18':
        sparse_features = ['user_id', 'user_gender_id', 'user_occupation_id', 'user_age_level', 'user_star_level',
                           'item_id', 'cate', 'item_brand_id', 'item_city_id', 'item_price_level', 'item_sales_level',
                           'item_collected_level', 'item_pv_level',
                           'shop_id', 'shop_review_num_level', 'shop_star_level',
                           'context_page_id', 'hour', 'dayhour', 'query1', 'same_cate', 'same_property'
                           ]
        var_features = []

        dense_features = ['shop_review_positive_rate', 'shop_score_service', 'shop_score_delivery',
                          'shop_score_description']
        labels = ['is_trade']
        data=pd.read_csv('/home/featurize/data/ijcai2018.csv')
        #domain_col='user_gender_id'
        #data[domain_col]-=1
        data = data[data[domain_col]>0]
        print(data[domain_col].value_counts())
        train_all = data[data['day']<7]
        print(train_all.shape[0])
        test_all = data[data['day']==7]
        print(test_all.shape[0])
        num_domains = data[domain_col].nunique()


    else:
        raise NotImplementedError('not implemented')





