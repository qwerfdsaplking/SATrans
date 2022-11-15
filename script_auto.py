
import pandas as pd
import os

#alicpp Namespace(att_head_num=4, att_layer_num=3, att_layer_type='trans', data_name='alicpp', domain_att_layer_num=0, domain_col='301', embedding_dim=32, filter_feats=False, flag='new', learning_rate=0.01, merge='no', model_name='AutoInt', num_query_bases=3, seed='1024', share_domain_dnn_across_layers=False)

#alicpp,121 cateid, 122, cate_group_id, 125 age id, 128 occupation, 129 geography
#cms_segid 97 #cms_group_id 13 #final_gender_code 3 #age_level 7 #pvalue_level 4 #shopping_level 4 #occupation 2 #new_user_class_level 5
domain_cols_dict = {
    'alimama':['age_level','pid','cms_group_id'],#,'cms_segid'],
    'alicpp':['122','124','125','126','127','128','129','301'],
    'ijcai18':['user_gender_id','user_age_level','user_occupation_id','user_star_level','context_page_id']
    #'ijcai18': ['user_gender_id', 'user_age_level', 'user_occupation_id']#, 'user_star_level']

}
domain_cols_dict = {'alimama':['pid'],'alicpp':['301'],'ijcai':['context_id'],'alicpp_filtered':['301']}


data_name_list = []
#data_name_list += ['alimama']
data_name_list += ['alicpp']  # ,'alicpp']#'alimama']'Star_Net',
#data_name_list += ['ijcai18']
#data_name_list += ['alicpp_filtered']

model_name_list = []
#,'Star_Trans_w_Out']'Star_Trans_w_Out',
#model_name_list += ['WDL','AutoInt','DCN','DeepFM']#,'DCN','DeepFM','AutoInt','NFM']#STAR_Trans
#model_name_list += ['Star_Net_unorm']
#model_name_list += ['Star_Net']

#model_name_list += ['Star_Trans']

#model_name_list += ['Starv2_Trans']
model_name_list += ['Starv3_Trans']
#model_name_list += ['AutoInt']

#model_name_list = ['SharedBottom','MMOE','PLE']
#model_name_list +=['Gate_Trans']
#flag='norelu'

flag_list = ['norelu','norelu-large']#['norelu-aggmap']
#flag_list = ['norelu-layerid']#['norelu-aggmap']
#flag_list = ['norelu-layerid-qkvid-cat']#['norelu-aggmap']
flag_list=['trans']
flag_list=['nometanorm']

flag_list = ['norelu']
flag_list = ['norelu-layerid-qkvid-noqkmap-cat','norelu-layerid-qkvid-noqkmap-sum']
flag_list = ['norelu-layerid-qkvid-noqkmap-novmap-cat','norelu-layerid-qkvid-noqkmap-novmap-sum']
flag_list = ['norelu-metabias','norelu']
flag_list = ['noreluxxx64-32','noreluxxx64-64-32','norelu-layerid-qkvid-noqkmap-catxxx64-32','norelu-layerid-qkvid-noqkmap-catxxx64-64-32']
#flag_list = ['norelu-largexxx64-64-32','norelu-large-layerid-qkvid-noqkmap-catxxx64-64-32']
flag_list = ['norelu','norelu-layerid-qkvid-noqkmap-cat','norelu-layerid-qkvid-noqkmap-sum']#,'norelu-layerid-qkvid-noqkmap-sum']
#flag_list = ['norelu','norelu-layerid-qkvid-noqkmap-cat']
flag_list=['classic']
#flag_list = ['norelu-layerid-qkvid-noqkmap-cat','norelu-layerid-qkvid-noqkmap-sum']
flag_list = ['norelu-layerid-qkvid-noqkmap-cat-nometanorm','norelu-layerid-qkvid-noqkmap-sum-nometanorm']
flag_list = ['norelu-nometanorm']
flag_list = ['norelu','norelu-layerid-qkvid-cat','norelu-times2']
flag_list = ['norelu-layerid-qkvid-noqkmap-cat','norelu-layerid-qkvid-noqkmap-sum']
flag_list = ['multi-task']
#flag_list = ['norelu-noattndrop','norelu-autoint']
flag_list = ['norelu-frelu','norelu']
flag_list = ['norelu-qkvdiffnorm']
flag_list = ['norelu-layerid-qkvid-noqkmap-sum-qkvdiffnorm']
flag_list = ['testdomain']
flag_list = ['norelu-qkvdiffnorm']
flag_list = ['norelu-bilinear']
flag_list = ['norelu-bilinear-layerid-qkvid']
flag_list = ['norelu-nometadrop','norelu-nometares','norelu-nometadrop-nometares']
flag_list=['norelu']
#flag_list = ['usetranslayers']
#seed_list = [1028,1029,1030,1031]#,1026,1027,1028]#,1026,1027]#,1026]#,1028]#[1025,1027]
seed_list = [1028]#,1028]
#seed_list=[1026]
embedding_dim_list=[32]

learning_rate_list=[0.005]#,0.001]#[0.01,0.001,0.005,0.002,0.0005,0.0001]
merge_list=['no']#['qkno','no']#['sum', 'dot', 'no']
att_layer_num_list=[3]
att_head_num_list = [4]
att_layer_type_list = ['deepctr']
num_query_bases_list = [4]#,5,6,7,3]
share_domain_dnn_across_layers_list = [True]#[True]
domain_att_layer_num_list = [3]#4
attn_batch_reg_list=[1.0]#[1.0,0.1,0.01,0.001,0.0001]
meta_mode_list=['QK']

from itertools import product

#如添加参数，修改此处
combines = [flag_list,seed_list,model_name_list,embedding_dim_list,learning_rate_list,#basic paramters
            merge_list,#star_trans
            att_layer_num_list,att_head_num_list,att_layer_type_list,#autoint
            num_query_bases_list,share_domain_dnn_across_layers_list,domain_att_layer_num_list,attn_batch_reg_list,meta_mode_list]#starv2_trans



f=open('./start.sh','w')
df_list=[]



for data_name in data_name_list:
    name_set = set()

    if False and os.path.exists(f'{data_name}_results.csv'):
        df = pd.read_csv(f'{data_name}_results.csv',header=None)
        df_list.append(df)
        insts_runned = df[0].tolist()
    else:
        insts_runned=[]

    for domain_col in domain_cols_dict[data_name]:
        for parameters in product(*combines):
            flag,seed, model_name, embedding_dim, learning_rate,merge, \
            att_layer_num, att_head_num, att_layer_type,\
            num_query_bases, share_domain_dnn_across_layers, domain_att_layer_num,attn_batch_reg,meta_mode = parameters#如添加参数，修改此处

            if data_name=='alicpp':
                learning_rate=0.005
            if data_name=='alimama':
                learning_rate=0.001

            cmd = f'python zmdr.py --data_name {data_name} --model_name {model_name} --seed {seed} --embedding_dim {embedding_dim} --learning_rate {learning_rate}'
            inst_name = f'{model_name}_{embedding_dim}_{learning_rate}_'
            if 'Star_Trans' in model_name:#todo
                inst_name += f'{domain_att_layer_num}_{att_head_num}_{merge}'
                cmd += f' --merge {merge} --domain_att_layer_num {domain_att_layer_num} --att_head_num {att_head_num} '
            elif 'Starv3_Trans' in model_name or 'Gate_Trans' in model_name:
                inst_name += f'{domain_att_layer_num}_{att_head_num}_{meta_mode}'
                cmd += f' --domain_att_layer_num {domain_att_layer_num} --att_head_num {att_head_num} --meta_mode {meta_mode}'

            elif 'Starv2_Trans' in model_name:#todo
                inst_name += f'{domain_att_layer_num}_{att_head_num}_{num_query_bases}_{share_domain_dnn_across_layers}_{attn_batch_reg} '
                cmd += f' --num_query_bases {num_query_bases} --share_domain_dnn_across_layers {share_domain_dnn_across_layers} --domain_att_layer_num {domain_att_layer_num} --att_head_num {att_head_num} --attn_batch_reg {attn_batch_reg} '
            elif 'AutoInt' in model_name:
                inst_name += f'{att_layer_num}_{att_head_num}_{att_layer_type}'
                cmd += f' --att_layer_num {att_layer_num} --att_head_num {att_head_num} --att_layer_type {att_layer_type} '
            else:
                pass
            inst_name += f'_{seed}_{domain_col}_{flag}'
            cmd += f' --domain_col {domain_col} --flag {flag}\n'
            #不在已有的结果里，不在不需要的组合里
            if inst_name not in name_set and inst_name not in insts_runned:
                f.write(cmd)
                name_set.add(inst_name)
            #print(cmd)


f.close()

f=open('./start.sh','r')
content=f.read()
f.close()
print(content)

for df in df_list:
    df['name'] = df[0].map(lambda x:'_'.join(x.split('_')[:-1]))
    df['seed'] = df[0].map(lambda x:x.split('_')[-1])
    df = df.sort_values(by=['seed','name'])
    dfm = df.groupby('name').mean().round(4)

    if dfm.columns.shape[0]==3:
        dfm.columns = ['All','domain_1','domain_2']
    elif dfm.columns.shape[0]==4:
        dfm.columns = ['All','domain_1','domain_2','domain_3']

    dfm = dfm.reset_index(drop=False)
    dfm = dfm[dfm['name'].map(lambda x:'Out' not in x)]

    #print(dfm)
    print()



