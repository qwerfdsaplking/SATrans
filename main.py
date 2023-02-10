import os.path
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names, VarLenSparseFeat
from argparse import ArgumentParser
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import warnings
from utils import *
from models.dcn import DCN
from models.pnn import PNN
from models.deepfm import DeepFM
from models.autoint import AutoInt
from models.xdeepfm import xDeepFM
from models.fibinet import FiBiNET
from models.afm import AFM
from models.nfm import NFM
from sklearn.metrics import roc_auc_score
from models.satrans import SATrans
from models.star import Star_Net
from models.sharedbottom import SharedBottom
from models.mmoe import MMOE
from models.ple import PLE
from models.esmm import ESMM
from models.wdl import WDL
from models.adasparse import AdaSparse
import time
warnings.filterwarnings('ignore')


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'




def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_name', type=str, default='alicpp')
    parser.add_argument('--model_name', type=str, default='SATrans')
    parser.add_argument('--seed', type=str, default=1024)
    parser.add_argument('--merge', type=str, default='no')
    parser.add_argument('--num_query_bases', type=int, default=3)
    parser.add_argument('--share_domain_dnn_across_layers', type=boolean_string, default=False)
    parser.add_argument('--domain_col', type=str, default='None')
    parser.add_argument('--embedding_dim', type=int, default=32)
    parser.add_argument('--att_layer_num', type=int, default=0)
    parser.add_argument('--domain_att_layer_num', type=int, default=3)
    parser.add_argument('--att_layer_type', type=str, default='deepctr')
    parser.add_argument('--att_head_num', type=int, default=4)
    parser.add_argument('--flag', type=str, default='None')
    parser.add_argument('--filter_feats', type=boolean_string, default=False)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--prompt', type=boolean_string, default=True)
    parser.add_argument('--finetune', type=boolean_string, default=False)
    parser.add_argument('--attn_batch_reg', type=float, default=0.1)
    parser.add_argument('--meta_mode', type=str, default='Query')
    args = parser.parse_args()
    return args




if __name__ == '__main__':
    print('starting...++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    args = parse_args()
    model_name = args.model_name
    data_name = args.data_name
    seed = args.seed
    merge = args.merge
    domain_col = args.domain_col
    embedding_dim = args.embedding_dim
    att_head_num = args.att_head_num
    att_layer_type = args.att_layer_type
    att_layer_num = args.att_layer_num
    domain_att_layer_num = args.domain_att_layer_num
    flag = args.flag
    batch_size = 4096 * 2
    test_batch_size = batch_size * 4
    learning_rate = args.learning_rate
    print(args)
    filter_feats = args.filter_feats
    meta_mode = args.meta_mode
    postfix = ''
    valid_cnt_per_epoch = 1

    #domain_col denotes scenario features
    default_domain_col_dict = {'alicpp': '301', 'alimama': 'pid'}
    if domain_col == 'None':
        domain_col = default_domain_col_dict[data_name.split('_')[0]]
    domain_col_list = args.domain_col.split('-')

    #
    if data_name == 'alicpp':
        # domain id starts from 1
        labels = ['click']
        sparse_features = ['101', '121', '122', '124', '125', '126', '127', '128', '129', '205', '206', '207', '210',
                           '216',
                           '508', '509', '702', '853', '301']
        # var_features = ['10914', '11014', '15014', '12714']
        var_features = []
        dense_features = []
        topk = 3
        train_all = get_aliccp_ctr_df('ctr_train' + postfix, labels + sparse_features + var_features, k=topk)
        print('load train finish')
        test_all = get_aliccp_ctr_df('ctr_test' + postfix, labels + sparse_features + var_features, k=topk)
        print('load test finish')


        if train_all['301'].min() == 0:
            train_all['301'] += 1
            test_all['301'] += 1
        if len(domain_col_list) == 1:
            print(pd.Series(train_all[domain_col]).value_counts())
            print(pd.Series(test_all[domain_col]).value_counts())





        # size of embedding matrices
        data_max = {'101': 444861, '121': 97, '122': 13, '124': 2, '125': 7, '126': 3, '127': 3, '128': 2, '129': 4,
                    '205': 4348615, '206': 8993, '207': 695124, '210': 99606, '216': 234880, '508': 8185,
                    '509': 472354,
                    '702': 167813, '853': 91358, '301': 3,
                    '10914': 12523, '11014': 2981271, '15014': 99555, '12714': 426101}

        if len(domain_col_list) == 1:
            num_domains = max(pd.Series(train_all[domain_col]).nunique(), data_max[domain_col])

        num_domains_list = [max(pd.Series(train_all[col]).nunique(), data_max[col]) for col in
                            domain_col_list]

    elif data_name == 'alimama':




        data = loadh52df('../../data/alimama.h5')
        labels = ['clk']
        sparse_features = ['user_id', 'adgroup_id', 'pid', 'cms_segid', 'cms_group_id', 'final_gender_code',
                           'age_level', 'pvalue_level', 'shopping_level', 'occupation', 'new_user_class_level',
                           'cate_id', 'campaign_id', 'customer', 'brand']
        var_features = []

        if 'sparseprice' in flag:# dense feature price as sparse feature
            print('transform price')
            sparse_features.append('price')
            dense_features=[]
            lbe = LabelEncoder()
            data['price'] = lbe.fit_transform(data['price'])
        else:
            dense_features = ['price']
            mms = MinMaxScaler(feature_range=(0, 1))
            data[dense_features] = mms.fit_transform(data[dense_features])

        if len(domain_col_list) == 1:
            print(data[domain_col].value_counts())

        split_time_stamp = '2017-05-12 00:00:00'
        ts = time.mktime(time.strptime(split_time_stamp, "%Y-%m-%d %H:%M:%S"))
        train_all = df2dict(data[data['time_stamp'] < ts])
        test_all = df2dict(data[data['time_stamp'] >= ts])
        data_max = dict()
        for key in data.columns:
            data_max[key] = data[key].max()
        if len(domain_col_list) == 1:
            num_domains = max(pd.Series(train_all[domain_col]).nunique(), data_max[domain_col])

        num_domains_list = [max(pd.Series(train_all[col]).nunique(), data_max[col]) for col in
                            domain_col_list]


    else:
        raise NotImplementedError('not implemented')



    #feature transformation of deepctr
    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=int(data_max[feat]) + 2, embedding_dim=embedding_dim)
                              for i, feat in enumerate(sparse_features)] + [DenseFeat(feat, 1)
                                                                            for feat in dense_features]
    varlen_feature_columns = [
        VarLenSparseFeat(SparseFeat(feat, vocabulary_size=data_max[feat] + 2, embedding_dim=embedding_dim), maxlen=topk,
                         combiner='max')
        for i, feat in enumerate(var_features)]
    linear_feature_columns = fixlen_feature_columns + varlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns + varlen_feature_columns
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        device = 'cuda:0'
        # device='cpu'



    if 'SATrans' in model_name:
        model_path=f'./checkpoints/{model_name}_{embedding_dim}_{learning_rate}_{domain_att_layer_num}_{att_head_num}_{meta_mode}_{seed}_{domain_col}_{flag}.pt'
    elif 'AutoInt' in model_name:
        model_path=f'./checkpoints/{model_name}_{embedding_dim}_{learning_rate}_{att_layer_num}_{att_head_num}_{att_layer_type}_{seed}_{domain_col}_{flag}.pt'
    else:
        model_path=f'./checkpoints/{model_name}_{embedding_dim}_{learning_rate}_{seed}_{domain_col}_{flag}.pt'





    if model_name in ['WDL', 'DCN', 'DeepFM', 'xDeepFM', 'NFM', 'AutoInt', 'AFM', 'FiBiNET', 'PNN','AdaSparse']:
        if model_name == 'WDL':
            MODEL = WDL
        if model_name == 'DCN':
            MODEL = DCN
        if model_name == 'DeepFM':
            MODEL = DeepFM
        if model_name == 'xDeepFM':
            MODEL = xDeepFM
            batch_size = 4096
            test_batch_size =4096*4
        if model_name == 'NFM':
            MODEL = NFM
        if model_name == 'AFM':
            MODEL = AFM
        if model_name == 'AutoInt':
            MODEL = AutoInt
        if model_name == 'xDeepFM':
            MODEL = xDeepFM
        if model_name == 'FiBiNET':
            MODEL = FiBiNET
        if model_name == 'AdaSparse':
            MODEL = AdaSparse
        if model_name != 'PNN':
            model = MODEL(linear_feature_columns=linear_feature_columns,
                          dnn_feature_columns=dnn_feature_columns,
                          seed=seed,
                          device=device,
                          domain_column=domain_col,
                          num_domains=num_domains,
                          flag=flag)
        if model_name == 'PNN':
            MODEL = PNN
            model = MODEL(
                dnn_feature_columns=dnn_feature_columns,
                seed=seed,
                device=device,
                domain_column=domain_col,
                num_domains=num_domains,
                flag=flag)
    elif  model_name in ['SharedBottom','MMOE','PLE','ESMM']:
        if model_name=='SharedBottom':
            MODEL = SharedBottom
        if model_name=='MMOE':
            MODEL = MMOE
        if model_name=='PLE':
            MODEL = PLE
        if model_name=='ESMM':
            MODEL=ESMM


        model =MODEL(dnn_feature_columns=dnn_feature_columns, seed=seed, device=device,
                     task_types=['binary']*num_domains,
                     task_names=['ctr%d' %(i) for i in range(num_domains)],domain_column=domain_col,flag=flag
                     )
    elif model_name in ['Star_Net']:
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
                         device=device,flag=flag)

    elif model_name == 'SATrans':
        att_layer_num = 0
        use_dnn = False
        use_linear = False
        use_domain_dnn_linear = False
        share_domain_dnn_across_layers = args.share_domain_dnn_across_layers
        attn_batch_reg = args.attn_batch_reg
        meta_mode = args.meta_mode

        model = SATrans(linear_feature_columns=linear_feature_columns,
                              dnn_feature_columns=dnn_feature_columns,
                              domain_column_list=domain_col_list,
                              num_domains_list=num_domains_list,
                              att_layer_num=att_layer_num,
                              domain_att_layer_num=domain_att_layer_num,
                              att_head_num=att_head_num,
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





    print(f'=============={data_name}=====================================================')
    print(f'model name: {model_name}..{flag}..{seed}...{domain_col}...====================================')
    print('===========================================================================')

    # Mix learning
    train = train_all
    test = test_all

    target = labels[0]
    train_model_input = {name: train[name] for name in sparse_features + dense_features + var_features}
    test_model_input = {name: test[name] for name in sparse_features + dense_features + var_features}
    train_labels = train[target]
    test_labels = test[target]

    epoch_num = 1
    if valid_cnt_per_epoch > 1 or epoch_num > 1:
        validation_data = (test_model_input, test_labels)
    else:
        validation_data = None

    if data_name=='alimama' and 'sparseprice' in flag:
        model.classes_ = lbe.classes_
    # model.cpu()

    if model_name in ['SharedBottom', 'MMOE', 'PLE', 'ESMM']:
        model.compile(torch.optim.Adam(model.parameters(), lr=learning_rate),
                      ["binary_crossentropy"] * num_domains,
                      metrics=["binary_crossentropy", 'auc'])
    else:
        model.compile(torch.optim.Adam(model.parameters(), lr=learning_rate), "binary_crossentropy",
                      metrics=["binary_crossentropy", 'auc'])
    model.fit(x=train_model_input,
              y=train_labels,
              validation_data=validation_data,
              valid_cnt_per_epoch=valid_cnt_per_epoch,
              batch_size=batch_size, epochs=epoch_num, verbose=1)



    pred_ans = model.predict(test_model_input, batch_size * 4)

    test_auc_list = []
    test_auc = round(roc_auc_score(test[target], pred_ans), 4)

    ####loss
    test_loss = F.binary_cross_entropy(torch.tensor(pred_ans).squeeze(),torch.tensor(test_labels).double()).item()

    test_auc_list.append(str(test_auc))
    print("test AUC", test_auc)




    domain_col_show = domain_col
    for i in range(test_model_input[domain_col_show].min(), test_model_input[domain_col_show].max() + 1):
        domain_indice = test_model_input[domain_col_show] == i
        domain_pred = pred_ans[domain_indice]
        domain_label = test_labels[domain_indice]
        domain_test_auc = round(roc_auc_score(domain_label, domain_pred), 4)
        print(f"Domain {i} test AUC", domain_test_auc)
        test_auc_list.append(str(domain_test_auc))

    from datetime import datetime

    dt = datetime.now().strftime('%m-%d-%H-%M')
    print(dt)

    file_name = f'./{data_name}_results.csv'
    f = open(file_name, 'a')
    if 'Star_Trans' in model_name:
        res = f'{dt}-{model_name}_{embedding_dim}_{learning_rate}_{domain_att_layer_num}_{att_head_num}_{merge}_{seed}_{domain_col}_{flag},' + ','.join(
            test_auc_list)+','+'%.6f' %test_loss
    elif 'SATrans' in model_name:
        res = f'{dt}-{model_name}_{embedding_dim}_{learning_rate}_{domain_att_layer_num}_{att_head_num}_{meta_mode}_{seed}_{domain_col}_{flag},' + ','.join(
            test_auc_list)+','+ '%.6f' %test_loss
    elif 'AutoInt' in model_name:
        res = f'{dt}-{model_name}_{embedding_dim}_{learning_rate}_{att_layer_num}_{att_head_num}_{att_layer_type}_{seed}_{domain_col}_{flag},' + ','.join(
            test_auc_list)+','+'%.6f' %test_loss
    else:
        res = f'{dt}-{model_name}_{embedding_dim}_{learning_rate}_{seed}_{domain_col}_{flag},' + ','.join(test_auc_list)+','+'%.6f' %test_loss
    f.write(res + '\n')
    f.close()


    torch.cuda.empty_cache()
    if 'dump' in flag:
        torch.save(model.cpu().state_dict(), model_path.replace('_instattn',''))
        dump_pkl(pred_ans,model_path.replace('.pt','_testpred.pkl'))










