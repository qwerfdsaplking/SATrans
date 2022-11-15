import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from deepctr_torch.models import *
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names,VarLenSparseFeat
from argparse import ArgumentParser
import seaborn
import warnings
warnings.filterwarnings('ignore')
from utils import *
from torch import nn
from models.star_trans import *
from models.dcn import *
from models.deepfm import *
from models.autoint import *
import copy
from models.sharedbottom import SharedBottom
from models.mmoe import MMOE
from models.ple import PLE

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
    parser.add_argument('--att_layer_type',type=str,default='deepctr')
    parser.add_argument('--att_head_num',type=int,default=2)
    parser.add_argument('--flag',type=str,default='None')
    parser.add_argument('--filter_feats',type=boolean_string,default=False)
    parser.add_argument('--learning_rate',type=float,default=0.001)
    parser.add_argument('--prompt',type=boolean_string,default=True)
    parser.add_argument('--finetune',type=boolean_string,default=False)
    parser.add_argument('--attn_batch_reg',type=float,default=0.1)
    parser.add_argument('--meta_mode',type=str,default='Query')


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
    h5_path = '/home/featurize/work/data/alicpp.h5'
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
    h5_path = '/home/featurize/work/data/alicpp.h5'
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
    h5_path = '/home/featurize/work/data/alicpp.h5'
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
    learning_rate=args.learning_rate
    print(args)
    filter_feats = args.filter_feats
    postfix=''
    valid_cnt_per_epoch=1


    default_domain_col_dict = {'alicpp':'301','alimama':'pid','ijcai18':'user_gender_id'}
    if domain_col == 'None':
        domain_col = default_domain_col_dict[data_name.split('_')[0]]



    if data_name == 'alicpp' or data_name=='alicpp_filtered':
        labels = ['click']
        sparse_features = ['101', '121', '122', '124', '125', '126', '127', '128', '129', '205', '206', '207', '210', '216',
                           '508', '509', '702', '853', '301']
        #var_features = ['10914', '11014', '15014', '12714']
        var_features = []
        dense_features = []
        topk=3
        if filter_feats or data_name=='alicpp_filtered':
            postfix='_filtered'
            print('000')
        train_all = get_ctr_df('ctr_train'+postfix,labels+sparse_features+var_features,k=topk)
        print('load train finish')
        test_all = get_ctr_df('ctr_test'+postfix,labels+sparse_features+var_features,k=topk)
        print('load test finish')

        #for key in train_all.keys():
        #    print(key,pd.Series(train_all[key]).nunique(),train_all[key].max())

        if train_all[domain_col].min()==0:
            train_all[domain_col]+=1
            test_all[domain_col]+=1
        print(pd.Series(train_all[domain_col]).value_counts())
        print(pd.Series(test_all[domain_col]).value_counts())



        data_max = {'101': 444861, '121': 97, '122': 13, '124': 2, '125': 7, '126': 3, '127': 3, '128': 2, '129': 4,
                    '205': 4348615, '206': 8993, '207': 695124, '210': 99606, '216': 234880, '508': 8185,
                    '509': 472354,
                    '702': 167813, '853': 91358, '301': 3,
                    '10914': 12523, '11014': 2981271, '15014': 99555, '12714': 426101}
        #domain_cols, did_map = get_domain_feat(train_all, domain_col, None)
        #domain_cols, did_map = get_domain_feat(test_all, domain_col, did_map)
        num_domains = pd.Series(train_all[domain_col]).nunique()

    elif data_name=='alimama':
        labels = ['clk']
        sparse_features = ['user_id','adgroup_id', 'pid','userid', 'cms_segid', 'cms_group_id', 'final_gender_code', 'age_level', 'pvalue_level', 'shopping_level', 'occupation', 'new_user_class_level', 'cate_id', 'campaign_id', 'customer', 'brand']
        var_features = []
        dense_features = ['price']
        data = pd.read_csv('~/work/data/alimama.csv')

        #domain_col = 'pid'
        if data[domain_col].min()==0:
            data[domain_col]+=1
        print(data[domain_col].value_counts())
        mms = MinMaxScaler(feature_range=(0, 1))
        data[dense_features] = mms.fit_transform(data[dense_features])
        import time
        split_time_stamp = '2017-05-12 00:00:00'
        ts = time.mktime(time.strptime(split_time_stamp, "%Y-%m-%d %H:%M:%S"))
        train_all = df2dict(data[data['time_stamp'] < ts])
        test_all = df2dict(data[data['time_stamp'] >= ts])
        num_domains = int(data[domain_col].max())
        data_max = dict()
        for key in data.columns:
            data_max[key] = data[key].max()

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
        data=pd.read_csv('/home/featurize/data/ijcai2018_v2.csv')
        #domain_col='user_gender_id'
        #data[domain_col]-=1
        print('before filter nan',data.shape[0])
        data = data[data[domain_col]>0]#去掉缺失值，缺失值在预处理中变为0
        print('after filter nan',data.shape[0])

        print(data[domain_col].value_counts())
        train_all = data[data['day']<7]
        print(train_all.shape[0])
        test_all = data[data['day']==7]
        print(test_all.shape[0])
        train_all = df2dict(train_all)
        test_all = df2dict(test_all)
        num_domains = data[domain_col].nunique()
        data_max = dict()
        for key in data.columns:
            data_max[key] = data[key].max()


    else:
        raise NotImplementedError('not implemented')

    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=int(data_max[feat])+2, embedding_dim=embedding_dim)
                              for i, feat in enumerate(sparse_features)] + [DenseFeat(feat, 1)
                                                      for feat in dense_features]
    varlen_feature_columns = [VarLenSparseFeat(SparseFeat(feat, vocabulary_size=data_max[feat]+2, embedding_dim=embedding_dim),maxlen=topk,combiner='max')
                              for i, feat in enumerate(var_features)]

    linear_feature_columns = fixlen_feature_columns+varlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns+varlen_feature_columns
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        device = 'cuda:0'
        #device='cpu'

    if model_name in ['WDL', 'DCN', 'DeepFM', 'xDeepFM', 'NFM']:
        if model_name == 'WDL':
            MODEL = WDL
        if model_name == 'DCN':
            MODEL = DCN
        if model_name == 'DeepFM':
            MODEL = DeepFM
        if model_name == 'xDeepFM':
            MODEL = xDeepFM
        if model_name == 'NFM':
            MODEL = NFM
        model = MODEL(linear_feature_columns=linear_feature_columns,
                      dnn_feature_columns=dnn_feature_columns,
                      seed=seed,
                      device=device,
                      )
    elif  model_name in ['SharedBottom','MMOE','PLE']:
        if model_name=='SharedBottom':
            MODEL = SharedBottom
        if model_name=='MMOE':
            MODEL = MMOE
        if model_name=='PLE':
            MODEL = PLE

        model =MODEL(dnn_feature_columns=dnn_feature_columns, seed=seed, device=device,
                     task_types=['binary']*num_domains,
                     task_names=['ctr%d' %(i+1) for i in range(num_domains)],domain_column=domain_col
                     )
    elif model_name in ['Star_Net', 'Star_Net_unorm']:
        use_domain_dnn = True
        if model_name == 'Star_Net':
            use_domain_bn = True
        else:
            use_domain_bn = False
        model = Star_Net(linear_feature_columns=linear_feature_columns,
                         dnn_feature_columns=dnn_feature_columns,
                         domain_column=domain_col,
                         num_domains=num_domains,
                         domain_id_as_feature=True,
                         dnn_hidden_units=(256, 128),
                         use_domain_dnn=use_domain_dnn,
                         use_domain_bn=use_domain_bn,
                         seed=seed,
                         device=device,
                         flag=flag)
    elif model_name in ['AutoInt', 'Star_Trans', 'Star_Trans_mixed', 'Star_Trans_w_Out']:
        if model_name == 'AutoInt':
            domain_att_layer_num = 0
            use_dnn = False
            use_linear = False
            use_domain_dnn_linear = False
        elif model_name == 'Star_Trans':
            att_layer_num = 0
            use_dnn = False
            use_linear = False
            use_domain_dnn_linear = False
        else:
            raise NotImplementedError('xx')

        model = Star_Trans(linear_feature_columns=linear_feature_columns,
                           dnn_feature_columns=dnn_feature_columns,
                           domain_column=domain_col,
                           num_domains=num_domains,
                           domain_id_as_feature=True,
                           att_layer_num=att_layer_num,
                           domain_att_layer_num=domain_att_layer_num,
                           att_head_num=att_head_num,
                           merge=merge,
                           att_layer_type=att_layer_type,
                           use_domain_dnn_linear=use_domain_dnn_linear,
                           use_linear=use_linear,
                           use_dnn=use_dnn,
                           seed=seed,
                           flag=flag,
                           device=device)
    elif model_name in ['Starv2_Trans']:
        att_layer_num = 0
        num_query_bases = args.num_query_bases
        use_dnn = False
        use_linear = False
        use_domain_dnn_linear = False
        share_domain_dnn_across_layers = args.share_domain_dnn_across_layers
        attn_batch_reg=args.attn_batch_reg
        # print(args)

        model = Starv2_Trans(linear_feature_columns=linear_feature_columns,
                             dnn_feature_columns=dnn_feature_columns,
                             domain_column=domain_col,
                             num_domains=num_domains,
                             num_query_bases=num_query_bases,
                             domain_id_as_feature=True,
                             att_layer_num=att_layer_num,
                             domain_att_layer_num=domain_att_layer_num,
                             att_head_num=att_head_num,
                             att_layer_type=att_layer_type,
                             attn_batch_reg=attn_batch_reg,
                             share_domain_dnn_across_layers=share_domain_dnn_across_layers,
                             use_domain_dnn_linear=use_domain_dnn_linear,
                             use_linear=use_linear,
                             use_dnn=use_dnn,
                             seed=seed,
                             device=device)
    elif model_name=='Starv3_Trans':
        att_layer_num = 0
        use_dnn = False
        use_linear = False
        use_domain_dnn_linear = False
        share_domain_dnn_across_layers = args.share_domain_dnn_across_layers
        attn_batch_reg=args.attn_batch_reg
        meta_mode=args.meta_mode
        # print(args)

        model = Starv3_Trans(linear_feature_columns=linear_feature_columns,
                             dnn_feature_columns=dnn_feature_columns,
                             domain_column=domain_col,
                             num_domains=num_domains,
                             domain_id_as_feature=True,
                             att_layer_num=att_layer_num,
                             domain_att_layer_num=domain_att_layer_num,
                             att_head_num=att_head_num,
                             att_layer_type=att_layer_type,
                             share_domain_dnn_across_layers=share_domain_dnn_across_layers,
                             use_domain_dnn_linear=use_domain_dnn_linear,
                             use_linear=use_linear,
                             meta_mode=meta_mode,
                             use_dnn=use_dnn,
                             seed=seed,
                             device=device,
                             flag=flag)
    elif model_name=='Gate_Trans':
        att_layer_num = 0
        use_dnn = False
        use_linear = False
        use_domain_dnn_linear = False
        share_domain_dnn_across_layers = args.share_domain_dnn_across_layers
        attn_batch_reg=args.attn_batch_reg
        meta_mode=args.meta_mode
        # print(args)

        model = Gate_Trans(linear_feature_columns=linear_feature_columns,
                             dnn_feature_columns=dnn_feature_columns,
                             domain_column=domain_col,
                             num_domains=num_domains,
                             domain_id_as_feature=True,
                             att_layer_num=att_layer_num,
                             domain_att_layer_num=domain_att_layer_num,
                             att_head_num=att_head_num,
                             att_layer_type=att_layer_type,
                             share_domain_dnn_across_layers=share_domain_dnn_across_layers,
                             use_domain_dnn_linear=use_domain_dnn_linear,
                             use_linear=use_linear,
                             meta_mode=meta_mode,
                             use_dnn=use_dnn,
                             seed=seed,
                             device=device,
                             flag=flag)
    else:
        raise ValueError('no such model')



    if False:
        print('expanding')
        train_idx_list=[]
        index = np.array(range(train_all[domain_col].shape[0]))
        max_n = 0
        for domain_id in range(1,num_domains+1):
            train_all_i = index[train_all[domain_col]==domain_id]
            train_idx_list.append(train_all_i)
            print(train_all_i.shape)
            max_n = max(max_n,train_all_i.shape[0])
            train_all_i = np.random.shuffle(train_all_i)

        weights_list=[]
        for i,idx in enumerate(train_idx_list):
            nidx = idx.shape[0]
            if nidx!=max_n:
                times = max_n//nidx+1
                idx = np.concatenate([idx for i in range(times)])[:max_n]
            train_idx_list[i]=idx
            print(idx.shape)
            weights = np.ones(idx.shape)*(nidx/max_n)
            weights_list.append(weights)
        weights=np.concatenate(weights_list)
        train_idx = np.concatenate(train_idx_list)

        for i,k in enumerate(train_all.keys()):
            train_all[k]=train_all[k][train_idx]
        train_all['weight']=weights
        test_all['weight']=np.ones(test_all[domain_col].shape[0])

    print(f'=============={data_name}=====================================================')
    print(f'model name: {model_name}..{flag}..{seed}....====================================')
    print('===========================================================================')

        #print('context %s'%(i))

    # Mix learning
    train=train_all
    test=test_all


    target = labels[0]
    train_model_input = {name: train[name] for name in sparse_features + dense_features + var_features}
    test_model_input = {name: test[name] for name in sparse_features + dense_features + var_features}
    train_labels = train[target]
    test_labels = test[target]

    #for key in train_model_input.keys():
    #    print(key, max(train_model_input[key].max(),test_model_input[key].max()))

    epoch_num = 1
    #if train[target].shape[0]<2000000:
    #    epoch_num=10
    if valid_cnt_per_epoch>1:
        validation_data = (test_model_input, test_labels)
    else:
        validation_data = None

    if model_name in ['SharedBottom','MMOE','PLE']:
        model.compile(torch.optim.Adam(model.parameters(), lr=learning_rate),
                      ["binary_crossentropy"]*num_domains,
                      metrics=["binary_crossentropy", 'auc'])
    else:


        model.compile(torch.optim.Adam(model.parameters(), lr=learning_rate), "binary_crossentropy",
                  metrics=["binary_crossentropy", 'auc'])
    model.fit(x=train_model_input,
              y=train_labels,
              validation_data=validation_data,
              valid_cnt_per_epoch=valid_cnt_per_epoch,
              batch_size=batch_size, epochs=epoch_num, verbose=1)

    pred_ans = model.predict(test_model_input, batch_size*4)
    #model.evaluate(test_model_input, test[target],4096*16)
    #print("test LogLoss", round(log_loss(test[target], pred_ans), 4))
    test_auc_list = []
    test_auc = round(roc_auc_score(test[target], pred_ans), 4)
    test_auc_list.append(str(test_auc))
    print("test AUC", test_auc)
    for i in range(num_domains):
        domain_indice = test_model_input[domain_col]==(i+1)
        domain_pred = pred_ans[domain_indice]
        domain_label = test_labels[domain_indice]
        test_auc = round(roc_auc_score(domain_label, domain_pred), 4)
        print(f"Domain {i+1} test AUC", test_auc)
        test_auc_list.append(str(test_auc))

    if False:
        self = model
        X = self.X
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        domain_ids = X[:, self.feature_index[self.domain_column][0]].long()

        if self.use_linear:
            logit = self.linear_model(X)  # 没考虑domain as feature
        else:
            logit = 0
        # logit = 0
        att_input = concat_fun(sparse_embedding_list, axis=1)

        # weight layers
        domain_emb = self.domain_embeddings(domain_ids)
        domain_emb = F.relu(domain_emb)
        query_weights_list = [F.softmax(weight_dnn(domain_emb), -1) for weight_dnn in self.domain_weights_DNNs]

        attn_reg_list = [self.attn_regularization_loss(weights, domain_ids).reshape(1) for weights in
                         query_weights_list]
        attn_reg = torch.cat(attn_reg_list).mean()
        print(query_weights_list[0][domain_ids == 1][:5])
        print(query_weights_list[0][domain_ids == 2][:5])
        print(query_weights_list[0][domain_ids == 3][:5])




    if args.prompt and False:
        finetune_test_auc_list = []
        for domain_id in range(1,num_domains+1):
            train=dict()
            test=dict()
            for k in train_all.keys():
                train_mask = train_all[domain_col]==domain_id
                test_mask = test_all[domain_col]==domain_id
                train[k] = train_all[k][train_mask]
                test[k] = test_all[k][test_mask]


            target = labels[0]
            train_model_input = {name: train[name] for name in sparse_features + dense_features + var_features}
            test_model_input = {name: test[name] for name in sparse_features + dense_features + var_features}
            train_labels = train[target]
            test_labels = test[target]


            #fix some parameters
            sub_model = copy.deepcopy(model)
            #for k, v in sub_model.named_parameters():
            #    if v.requires_grad:
             #       print(k)
            if args.prompt and model_name=='Starv2_Trans':
                trainable_set = ['domain_weights_DNNs', 'domain_embeddings']
                for k,v in sub_model.named_parameters():
                    flag=False
                    for name in trainable_set:
                        if name in k:
                            flag = True
                            break
                    v.requires_grad = True if flag else False

                for k, v in sub_model.named_parameters():
                    if v.requires_grad:
                        print(k)



            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, sub_model.parameters()), lr=learning_rate)
            sub_model.compile(optimizer, "binary_crossentropy",
                          metrics=["binary_crossentropy", 'auc'])

            epoch_num = 2
            valid_cnt_per_epoch = 5
            if valid_cnt_per_epoch > 1:
                validation_data = (test_model_input, test_labels)
            else:
                validation_data = None
            sub_model.fit(x=train_model_input,
                      y=train_labels,
                      validation_data=validation_data,
                      valid_cnt_per_epoch=valid_cnt_per_epoch,
                      batch_size=batch_size, epochs=epoch_num, verbose=1)

            pred_ans = sub_model.predict(test_model_input, batch_size * 4)
            # model.evaluate(test_model_input, test[target],4096*16)
            # print("test LogLoss", round(log_loss(test[target], pred_ans), 4))
            test_auc = round(roc_auc_score(test[target], pred_ans), 4)
            print(f"Domain {domain_id} test AUC", test_auc)
            finetune_test_auc_list.append(str(test_auc))






    file_name = f'./{data_name}_results.csv'
    f = open(file_name,'a')
    if 'Star_Trans' in model_name:
        res = f'{model_name}_{embedding_dim}_{learning_rate}_{domain_att_layer_num}_{att_head_num}_{merge}_{seed}_{domain_col}_{flag},'+','.join(test_auc_list)
    elif 'Starv3_Trans' in model_name:
        res = f'{model_name}_{embedding_dim}_{learning_rate}_{domain_att_layer_num}_{att_head_num}_{meta_mode}_{seed}_{domain_col}_{flag},'+','.join(test_auc_list)
    elif 'Starv2_Trans' in model_name:
        res = f'{model_name}_{embedding_dim}_{learning_rate}_{domain_att_layer_num}_{att_head_num}_{num_query_bases}_{share_domain_dnn_across_layers}_{attn_batch_reg}_{seed}_{domain_col}_{flag},'+','.join(test_auc_list)
    elif 'AutoInt' in model_name:
        res = f'{model_name}_{embedding_dim}_{learning_rate}_{att_layer_num}_{att_head_num}_{att_layer_type}_{seed}_{domain_col}_{flag},'+','.join(test_auc_list)
    else:
        res = f'{model_name}_{embedding_dim}_{learning_rate}_{seed}_{domain_col}_{flag},'+','.join(test_auc_list)
    f.write(res+'\n')
    f.close()

    torch.cuda.empty_cache()



    #=====================
    x=test_model_input
    self = model.eval()
    if isinstance(x, dict):
        x = [x[feature] for feature in self.feature_index]
    for i in range(len(x)):
        if len(x[i].shape) == 1:
            x[i] = np.expand_dims(x[i], axis=1)

    tensor_data = Data.TensorDataset(
        torch.from_numpy(np.concatenate(x, axis=-1)))
    test_loader = DataLoader(
        dataset=tensor_data, shuffle=False, batch_size=batch_size)

    pred_ans = []
    with torch.no_grad():
        for _, x_test in tqdm(enumerate(test_loader)):
            x = x_test[0].to(self.device).float()

            y_pred = model(x).cpu().data.numpy()  # .squeeze()

            attn_score = [layer.normalized_att_scores for layer in model.domain_int_layers]

            break











