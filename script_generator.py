
import pandas as pd
import os


domain_cols_dict = {
    'alimama':['user_id', 'adgroup_id', 'pid',  'cms_segid', 'cms_group_id', 'final_gender_code',
                           'age_level', 'pvalue_level', 'shopping_level', 'occupation', 'new_user_class_level',
                           'cate_id', 'campaign_id', 'customer', 'brand'],
    'alicpp':[ '101', '121', '122', '124', '125', '126', '127', '128', '129', '205', '206', '207', '210','216', '508', '509', '702', '853','301'],
}
domain_cols_dict = {'alimama':['shopping_level'],
                    'alicpp':['301']}


data_name_list = []
#data_name_list += ['alicpp']  # ,'alicpp']#'alimama']'Star_Net',
data_name_list += ['alimama']


model_name_list = []
#model_name_list += ['MMOE','PLE','SharedBottom']
#model_name_list += ['Star_Net','AdaSparse']
model_name_list += ['SATrans']

classic=0
if classic:
    model_name_list = []
    model_name_list += ['AFM']
    model_name_list += ['WDL']
    model_name_list+=['DCN']
    model_name_list +=['DeepFM']
    model_name_list += ['xDeepFM']
    model_name_list += ['NFM']
    model_name_list += ['PNN']
    model_name_list+=['AutoInt']




flag_list = ['mex-samemeta']#,'mex-pos']
flag_list = ['mex-pos-samemeta','mex-pos']
#flag_list = ['mex-samemeta-onlyemb','mex-samemeta','mex-pos']
flag_list = ['mex-samemeta-onlyemb-gate', 'mex-samemeta-gate','mex-pos-gate']
flag_list = ['mex-samemeta-onlyemb-bilinear','mex-samemeta-bilinear','mex-pos-bilinear']
flag_list = ['usetransfixed']
#flag_list=['mex-pos']
#flag_list = ['classic','usetrans']


flag_list = ['multitask']
flag_list = ['classic']
flag_list = ['classic_single']
flag_list = ['classic_finetune_decay']
#flag_list = ['mex-pos']#,'mex-pos-gate']
#flag_list = ['mex-nometanorm-samemeta-headnet','mex-samemeta-headnet']
#flag_list = ['mex-bilineartras']
#flag_list = ['mex-gatetrans']
#flag_list = ['mex-pos']#,'mex-samemeta-pos']
flag_list = ['sota']
#flag_list = ['sota-pos-sparseprice','sota-pos']


seed_list=[1000,1001]
seed_list=[1002,1003]
seed_list=[1004,1005]
seed_list = [1020,1021,1022,1023,1018,1019,1017,1016]#,1028]
seed_list = [1024,1025,1026,1027,1028,1029,1030,1031]#,1026,1027]#,1026]#,1028]#[1025,1027]
seed_list = [1024,1025,1026,1027,1028,1029,1030,1031]#,1026,1027]#,1026]#,1028]#[1025,1027]
seed_list = [1020,1021,1022,1023,1024,1025,1026,1027,1028]#alimama
seed_list = [1020,1021,1022,1023,1024,1025,1026,1027,1028]#alimama

seed_list = [1020,1021,1025,1028]#,1025,1028]
#seed_list = [1020,1021,1022,1023,1024]
seed_list = [1022,1023,1024,1026,1027]
seed_list = [1021,1022] #best alimama shopping level
seed_list = [1021,1022]
#seed_list = [1021,1022]#,1023]
seed_list = [1021]#,1021,1025,1028]#best alimama   mex-pos
seed_list = [1020,1021,1022,1023,1024,1025,1026,1027,1028,1029,1030] #best alimama shopping level
#seed_list = [1027]
embedding_dim_list=[32]

learning_rate_list=[0.001]#,0.001]#[0.01,0.001,0.005,0.002,0.0005,0.0001]
merge_list=['no']#['qkno','no']#['sum', 'dot', 'no']
att_layer_num_list=[3]
att_head_num_list = [4]
att_layer_type_list = ['deepctr']
num_query_bases_list = [4]#,5,6,7,3]
share_domain_dnn_across_layers_list = [True]#[True]
domain_att_layer_num_list = [3]#[1,2,3,4,5]#4
attn_batch_reg_list=[1.0]#[1.0,0.1,0.01,0.001,0.0001]
meta_mode_list=['QK']#['QKV','Q','K','V','QV','KV']
from itertools import product

#如添加参数，修改此处
combines = [seed_list,flag_list,model_name_list,embedding_dim_list,learning_rate_list,#basic paramters
            merge_list,#star_trans
            att_layer_num_list,att_head_num_list,att_layer_type_list,#autoint
            num_query_bases_list,share_domain_dnn_across_layers_list,domain_att_layer_num_list,attn_batch_reg_list,meta_mode_list]#starv2_trans



f=open('./start.sh','w')
df_list=[]


#
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
            seed, flag, model_name, embedding_dim, learning_rate,merge, \
            att_layer_num, att_head_num, att_layer_type,\
            num_query_bases, share_domain_dnn_across_layers, domain_att_layer_num,attn_batch_reg,meta_mode = parameters#如添加参数，修改此处

            file_name = 'main.py'
            if 'single' in flag:
                file_name = 'single_main.py'
            if 'finetune' in flag:
                file_name = 'finetune_main.py'
            if data_name=='avazu':
                embedding_dim=16
                pass
            elif data_name=='alicpp' and ('Trans' in model_name or 'usetrans' in flag):
                learning_rate=0.005 if learning_rate not in [0.005, 0.002] else learning_rate
                flag='sota'
            #if data_name=='alimama':
            else:
                #flag='sota-pos-sparseprice'

                learning_rate=0.001

            cmd = f'python {file_name} --data_name {data_name} --model_name {model_name} --seed {seed} --embedding_dim {embedding_dim} --learning_rate {learning_rate}'
            inst_name = f'{model_name}_{embedding_dim}_{learning_rate}_'
            if 'Star_Trans' in model_name:#todo
                inst_name += f'{domain_att_layer_num}_{att_head_num}_{merge}'
                cmd += f' --merge {merge} --domain_att_layer_num {domain_att_layer_num} --att_head_num {att_head_num} '
            elif 'SATrans' in model_name:
                inst_name += f'{domain_att_layer_num}_{att_head_num}_{meta_mode}'
                cmd += f' --domain_att_layer_num {domain_att_layer_num} --att_head_num {att_head_num} --meta_mode {meta_mode}'
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



#%run main.py --data_name alicpp --model_name SATrans --seed 1020 --embedding_dim 32 --learning_rate 0.005 --domain_att_layer_num 3 --att_head_num 4 --meta_mode QK --domain_col 301 --flag mex-samemeta-dump-showattn


